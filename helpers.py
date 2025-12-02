import re, json, math
from typing import List, Dict, Tuple

# Columns targeted by SQL Full-Text Search
SEARCH_COLUMNS = "(Title, LearningContent, ChallengesFaced, RecommendationOrLessonLearned)"

# Minimal English stopwords set (you can extend if needed)
STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","from","in","into","is","it","of",
    "on","or","s","such","t","that","the","their","then","there","these","they","this","to",
    "was","will","with","we","you","your","our","i","me","my","ours","yours"
}

# ------------------------- Query Processing -------------------------

def tokenize_query(q: str) -> Tuple[List[str], List[str]]:
    """
    Returns (phrases, tokens)
    - phrases: quoted phrases in the query (e.g., "sub surfaces")
    - tokens: remaining tokens (lowercased, stopwords removed, min length 3)
    """
    q = q.strip()
    # Extract quoted phrases
    phrases = re.findall(r'"([^"]+)"', q)
    # Remove phrases from q
    q_wo_phrases = re.sub(r'"[^"]+"', ' ', q)
    # Split rest on non-alphanum
    toks = [t.lower() for t in re.split(r'[^a-zA-Z0-9\-]+', q_wo_phrases) if t]
    # Keep tokens length>=3 and not stopwords
    toks = [t for t in toks if len(t) >= 3 and t not in STOPWORDS]
    # Normalize phrases
    phrases = [p.strip().lower() for p in phrases if p.strip()]
    return phrases, toks

def term_variants_for_contains(token: str) -> str:
    """
    Build an FTS OR group for a token with some robustness:
    - inflectional forms (nouns/verbs)
    - prefix match
    - hyphen/space alternates if token contains '-'
    """
    parts = []
    safe = token.replace('"', '')  # basic sanitize
    parts.append(f'FORMSOF(INFLECTIONAL, "{safe}")')
    parts.append(f'"{safe}*"')
    if '-' in safe:
        alt1 = safe.replace('-', ' ')
        alt2 = safe.replace('-', '')
        parts.append(f'"{alt1}*"')
        parts.append(f'"{alt2}*"')
    return "(" + " OR ".join(parts) + ")"

def phrase_for_contains(phrase: str) -> str:
    """
    Exact phrase + prefix; also add hyphen/space alternates.
    """
    p = phrase.replace('"', '')
    alts = [f'"{p}"', f'"{p}*"']
    if '-' in p:
        alts.append(f'"{p.replace("-", " ")}"')
        alts.append(f'"{p.replace("-", " ")}*"')
        alts.append(f'"{p.replace("-", "")}*"')
    return "(" + " OR ".join(alts) + ")"

def build_contains_predicates(query: str) -> Tuple[str, str, List[str]]:
    """
    Returns (exact_predicate, partial_predicate, count_terms)
    count_terms = list of raw terms/phrases we will count against content.
    """
    phrases, toks = tokenize_query(query)
    exact_parts: List[str] = []
    partial_parts: List[str] = []

    # Add phrases
    for ph in phrases:
        grp = phrase_for_contains(ph)
        exact_parts.append(grp)
        partial_parts.append(grp)

    # Add tokens
    for t in toks:
        grp = term_variants_for_contains(t)
        exact_parts.append(grp)
        partial_parts.append(grp)

    if not exact_parts and not partial_parts:
        # fallback when no meaningful terms; caller may switch to FREETEXT if desired
        return '""', '""', []

    # Exact: require all groups (AND)
    exact_pred = " AND ".join(exact_parts) if exact_parts else '""'
    # Partial: allow any (OR)
    partial_pred = " OR ".join(partial_parts) if partial_parts else '""'

    # Terms to count (frequency) — use phrases + tokens; add simple hyphen alternates for counting
    count_terms: List[str] = []
    for ph in phrases:
        count_terms.extend({ph, ph.replace("-", " "), ph.replace("-", "")})
    for t in toks:
        count_terms.extend({t, t.replace("-", " "), t.replace("-", "")})

    # De-duplicate while preserving insertion order
    seen = set()
    ct: List[str] = []
    for w in count_terms:
        if w and w not in seen:
            seen.add(w)
            ct.append(w)
    return exact_pred, partial_pred, ct

# ------------------------- Database Access -------------------------

def run_containstable(conn, predicate: str, top_n: int) -> List[Tuple[int, int]]:
    """
    Returns list of (ID, rank) for CONTAINSTABLE with given predicate.
    """
    if not predicate:
        return []
    rows: List[Tuple[int, int]] = []
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT TOP ({top_n}) ct.[KEY], ct.[RANK]
            FROM CONTAINSTABLE(dbo.Learnings, {SEARCH_COLUMNS}, ?) AS ct
            ORDER BY ct.[RANK] DESC, ct.[KEY] ASC
        """, (predicate,))
        for rid, rrank in cur.fetchall():
            rows.append((int(rid), int(rrank)))
    return rows

def fetch_docs(conn, ids: List[int]) -> Dict[int, Dict]:
    """
    Fetch doc fields for given IDs.
    """
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    sql = f"""
        SELECT ID, Title, LearningContent, ChallengesFaced, RecommendationOrLessonLearned
        FROM dbo.Learnings
        WHERE ID IN ({placeholders})
    """
    docs: Dict[int, Dict] = {}
    with conn.cursor() as cur:
        cur.execute(sql, ids)
        for r in cur.fetchall():
            ID, Title, LC, CF, RLL = r
            docs[int(ID)] = {
                "ID": int(ID),
                "Title": Title or "",
                "LearningContent": LC or "",
                "ChallengesFaced": CF or "",
                "RecommendationOrLessonLearned": RLL or ""
            }
    return docs

def fetch_all_embeddings(conn) -> Dict[int, List[float]]:
    """
    Assumes embeddings stored as JSON array in dbo.LearningsEmbeddings(EmbeddingJson).
    """
    embs: Dict[int, List[float]] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT ID, EmbeddingJson FROM dbo.LearningsEmbeddings")
        for ID, ej in cur.fetchall():
            try:
                vec = json.loads(ej) if ej else None
                if isinstance(vec, list) and vec:
                    embs[int(ID)] = [float(x) for x in vec]
            except Exception:
                # Skip malformed vectors
                continue
    return embs

# ------------------------- Text & Embedding Utils -------------------------

def combined_text(doc: Dict) -> str:
    return (" ".join([
        doc.get("Title",""), doc.get("LearningContent",""),
        doc.get("ChallengesFaced",""), doc.get("RecommendationOrLessonLearned","")
    ])).lower()

def count_occurrences(text: str, term: str) -> int:
    # Count whole word or phrase occurrences
    term = term.lower().strip()
    if not term:
        return 0
    if re.match(r"^[a-z0-9\- ]+$", term):
        # convert spaces to \s+ for phrases
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
    else:
        pattern = re.escape(term)
    return len(re.findall(pattern, text, flags=re.IGNORECASE))

def score_frequency(text: str, count_terms: List[str]) -> Tuple[int, Dict[str,int]]:
    per_term = {t: count_occurrences(text, t) for t in count_terms}
    total = sum(per_term.values())
    return total, per_term

def cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb + 1e-12)

def get_query_embedding(model, text: str) -> List[float]:
    """
    Embed the query using a local SentenceTransformers model instance.
    """
    vec = model.encode([text], normalize_embeddings=False)[0]
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)
