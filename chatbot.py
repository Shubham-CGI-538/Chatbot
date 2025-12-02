import os, pyodbc
from typing import List, Dict, Tuple, TypedDict
from dotenv import load_dotenv

# LangGraph modern API
from langgraph.graph import StateGraph, START, END

# Local embeddings (offline)
from sentence_transformers import SentenceTransformer

# Our helpers
from helpers import (
    build_contains_predicates,
    run_containstable,
    fetch_docs,
    combined_text,
    score_frequency,
    get_query_embedding,
    fetch_all_embeddings,
    cosine_sim
)

load_dotenv(override=True)

# ----- CONFIG -----
DB_CONN_STR = os.getenv("DB_CONN_STR")

# Local model directory (recommended to avoid network)
EMBED_MODEL_DIR = os.getenv("EMBED_MODEL_DIR")  # e.g., r"C:\models\all-MiniLM-L6-v2"
# Allow online download only if explicitly set (not recommended in corp TLS contexts)
ALLOW_HF_DOWNLOAD = os.getenv("ALLOW_HF_DOWNLOAD", "false").lower() == "true"

MAX_EXACT = int(os.getenv("MAX_EXACT", "200"))
MAX_PARTIAL = int(os.getenv("MAX_PARTIAL", "400"))
MAX_SEMANTIC = int(os.getenv("MAX_SEMANTIC", "100"))

if not DB_CONN_STR:
    raise RuntimeError("DB_CONN_STR is not set. Put your ODBC connection string in .env")

# ----- INIT RESOURCES -----
conn = pyodbc.connect(DB_CONN_STR)
conn.timeout = 60

# Load local embedding model directory (offline)
if not EMBED_MODEL_DIR and not ALLOW_HF_DOWNLOAD:
    raise RuntimeError(
        "EMBED_MODEL_DIR is not set and ALLOW_HF_DOWNLOAD=false.\n"
        "Set EMBED_MODEL_DIR to a local Sentence Transformers model directory "
        "(e.g., C:\\models\\all-MiniLM-L6-v2) to avoid network downloads."
    )

if EMBED_MODEL_DIR:
    model = SentenceTransformer(EMBED_MODEL_DIR)
else:
    # Fallback: allow HF download only when approved
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----- STATE & NODES -----

class ChatState(TypedDict):
    query: str
    exact_predicate: str
    partial_predicate: str
    count_terms: List[str]
    exact_hits: List[Tuple[int, int]]
    partial_hits: List[Tuple[int, int]]
    exact_results: List[Dict]
    partial_results: List[Dict]
    semantic_results: List[Dict]
    answers: List[str]

def refine_query_node(state: ChatState) -> ChatState:
    query = state["query"]
    exact_pred, partial_pred, count_terms = build_contains_predicates(query)
    state["exact_predicate"] = exact_pred
    state["partial_predicate"] = partial_pred
    state["count_terms"] = count_terms
    return state

def fts_search_node(state: ChatState) -> ChatState:
    exact_pred = state.get("exact_predicate", "")
    partial_pred = state.get("partial_predicate", "")
    exact_hits = run_containstable(conn, exact_pred, MAX_EXACT) if exact_pred else []
    partial_hits = run_containstable(conn, partial_pred, MAX_PARTIAL) if partial_pred else []
    state["exact_hits"] = exact_hits
    state["partial_hits"] = partial_hits
    return state

def rank_and_format_node(state: ChatState) -> ChatState:
    count_terms = state.get("count_terms", [])
    exact_hits = state.get("exact_hits", [])
    partial_hits = state.get("partial_hits", [])

    exact_ids = [i for i, _ in exact_hits]
    partial_ids = [i for i, _ in partial_hits if i not in set(exact_ids)]
    docs = fetch_docs(conn, exact_ids + partial_ids)

    # Tier 1: exact (sum of all term counts), tie-break by FTS rank
    exact_scored = []
    for (rid, ft_rank) in exact_hits:
        doc = docs.get(rid)
        if not doc:
            continue
        text = combined_text(doc)
        total, per_term = score_frequency(text, count_terms)
        exact_scored.append({
            "priority": 1,
            "ID": rid, "doc": doc,
            "total_count": total,
            "per_term": per_term,
            "ft_rank": ft_rank
        })
    exact_scored.sort(key=lambda r: (-r["total_count"], -r["ft_rank"], r["ID"]))

    # Tier 2: partial (strongest single term), tie-break by FTS rank
    exact_set = {x["ID"] for x in exact_scored}
    partial_scored = []
    for (rid, ft_rank) in partial_hits:
        if rid in exact_set:
            continue
        doc = docs.get(rid)
        if not doc:
            continue
        text = combined_text(doc)
        total, per_term = score_frequency(text, count_terms)
        strongest = max(per_term.values()) if per_term else 0
        partial_scored.append({
            "priority": 2,
            "ID": rid, "doc": doc,
            "total_count": total,
            "strongest_term_count": strongest,
            "per_term": per_term,
            "ft_rank": ft_rank
        })
    partial_scored.sort(key=lambda r: (-r["strongest_term_count"], -r["ft_rank"], r["ID"]))

    state["exact_results"] = exact_scored
    state["partial_results"] = partial_scored
    return state

def semantic_append_node(state: ChatState) -> ChatState:
    # Tier 3: semantic for items not in tiers 1-2
    seen_ids = {r["ID"] for r in state.get("exact_results", [])} | {r["ID"] for r in state.get("partial_results", [])}
    q = state["query"]
    q_emb = get_query_embedding(model, q)
    all_embs = fetch_all_embeddings(conn)

    scored: List[Tuple[int, float]] = []
    for rid, emb in all_embs.items():
        if rid in seen_ids:
            continue
        # Guard against dimension mismatch
        if not emb or len(emb) != len(q_emb):
            continue
        sim = cosine_sim(q_emb, emb)
        scored.append((rid, sim))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:MAX_SEMANTIC]
    docs = fetch_docs(conn, [i for i, _ in top])

    semantic_rows = []
    for rid, sim in top:
        if rid not in docs:
            continue
        semantic_rows.append({
            "priority": 3,
            "ID": rid,
            "doc": docs[rid],
            "semantic_score": sim
        })
    state["semantic_results"] = semantic_rows
    return state

def produce_answer_node(state: ChatState) -> ChatState:
    def fmt_row(r):
        d = r["doc"]
        preview = (d.get("LearningContent") or d.get("RecommendationOrLessonLearned") or d.get("ChallengesFaced") or d.get("Title") or "")[:300]
        if r["priority"] == 1:
            meta = f"total_term_hits={r['total_count']}, ft_rank={r['ft_rank']}"
        elif r["priority"] == 2:
            meta = f"strongest_term_hits={r['strongest_term_count']}, ft_rank={r['ft_rank']}"
        else:
            meta = f"semantic_score={r['semantic_score']:.4f}"
        return f"[P{r['priority']}] ID={d['ID']} | Title: {d['Title']}\n{preview}\n({meta})"

    ordered = (
        state.get("exact_results", []) +
        state.get("partial_results", []) +
        state.get("semantic_results", [])
    )
    state["answers"] = [fmt_row(r) for r in ordered] or ["No relevant data found."]
    return state

# ----- BUILD GRAPH -----
graph = StateGraph(ChatState)
graph.add_node("refine_query", refine_query_node)
graph.add_node("fts_search",   fts_search_node)
graph.add_node("rank_format",  rank_and_format_node)
graph.add_node("semantic",     semantic_append_node)
graph.add_node("produce",      produce_answer_node)

graph.add_edge(START, "refine_query")
graph.add_edge("refine_query", "fts_search")
graph.add_edge("fts_search", "rank_format")
graph.add_edge("rank_format", "semantic")
graph.add_edge("semantic", "produce")
graph.add_edge("produce", END)

app = graph.compile()

def chatbot(query: str) -> List[str]:
    state = app.invoke({"query": query})
    return state.get("answers", [])

# --- Add this to the end of chatbot.py ---

def _format_full_markdown(results: List[Dict], heading: str) -> str:
    if not results:
        return ""
    lines = [f"### {heading} ({len(results)})"]
    for r in results:
        d = r["doc"]
        lines.append(
f"""**ID:** {d.get('ID')}
**Title:** {d.get('Title')}

**LearningContent**
{d.get('LearningContent','').strip()}

**ChallengesFaced**
{d.get('ChallengesFaced','').strip()}

**RecommendationOrLessonLearned**
{d.get('RecommendationOrLessonLearned','').strip()}

---
""".rstrip()
        )
    return "\n".join(lines)

def chatbot_full(query: str) -> str:
    """
    Runs the pipeline and returns a single Markdown string with
    the FULL content of each entry (no truncation), grouped by priority.
    """
    state = app.invoke({"query": query})
    exact = state.get("exact_results", []) or []
    partial = state.get("partial_results", []) or []
    semantic = state.get("semantic_results", []) or []

    parts = []
    parts.append(_format_full_markdown(exact, "Tier 1 - Exact Matches"))
    parts.append(_format_full_markdown(partial, "Tier 2 - Partial Matches"))
    parts.append(_format_full_markdown(semantic, "Tier 3 - Semantic Matches"))

    md = "\n\n".join(p for p in parts if p).strip()
    return md if md else "_No relevant data found._"