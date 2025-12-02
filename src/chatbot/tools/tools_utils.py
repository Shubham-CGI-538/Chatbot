import pyodbc
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(override=True)

DB_CONN_STR = os.getenv("DB_CONN_STR")
if not DB_CONN_STR:
    raise RuntimeError("DB_CONN_STR not found in environment (.env).")

conn = pyodbc.connect(DB_CONN_STR, autocommit=True)


def get_relevant_ids(user_query: str) -> List[int]:
    """Map user query to relevant IDs using keyword search (multi-term)."""
    uq = (user_query or "").strip()
    if not uq:
        return []

    terms = [t.strip() for t in uq.split() if t.strip()]
    if not terms:
        return []

    conditions = []
    params = []
    for term in terms:
        like = f"%{term}%"
        conditions.append("(Title LIKE ? OR LearningContent LIKE ? OR ChallengesFaced LIKE ? OR RecommendationOrLessonLearned LIKE ?)")
        params.extend([like, like, like, like])

    sql = f"SELECT ID FROM dbo.Learnings WHERE {' AND '.join(conditions)}"
    # Using AND between term groups ensures all terms appear somewhere in the row
    ids: List[int] = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for row in cur.fetchall():
            ids.append(int(row[0]))

    return ids


def fetch_docs(ids: List[int]) -> Dict[int, Dict]:
    """Fetch document fields for given IDs from dbo.Learnings table."""
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


# tools_utils.py (add this function)
import re
from typing import List, Tuple

def _tokenize(text: str) -> List[str]:
    """Basic tokenizer: lowercases, removes punctuation/hyphens, splits on whitespace."""
    if not text:
        return []
    t = text.lower()
    t = re.sub(r"[-_/]", " ", t)     # treat hyphens as spaces: "sub-surface" -> "sub surface"
    t = re.sub(r"[^\w\s]", " ", t)   # strip punctuation
    tokens = [w for w in t.split() if w]
    return tokens

def get_relevant_ids_booleans(
    original_query: str,
    refined_query: str,
    min_should: int = 1
) -> List[int]:
    """
    Boolean matching across columns:
      - MUST: tokens from original_query (AND across terms)
      - SHOULD: tokens from refined_query that are not already in MUST (score threshold)
    Returns IDs that match all MUST terms and at least `min_should` SHOULD terms (if any).
    """

    must_terms = _tokenize(original_query)
    should_terms = [t for t in _tokenize(refined_query) if t not in must_terms]

    # If refiner adds many synonyms, keep min_should manageable
    if not should_terms:
        min_should = 0   # no optional terms to score

    # Build WHERE for MUST terms (AND of ORs across columns)
    must_conds = []
    params: List[str] = []
    for t in must_terms:
        like = f"%{t}%"
        must_conds.append(
            "(Title LIKE ? OR LearningContent LIKE ? OR ChallengesFaced LIKE ? OR RecommendationOrLessonLearned LIKE ?)"
        )
        params.extend([like, like, like, like])

    where_must = " AND ".join(must_conds) if must_conds else "1=1"

    # Build a score expression for SHOULD terms
    score_exprs = []
    score_params: List[str] = []
    for t in should_terms:
        like = f"%{t}%"
        score_exprs.append(
            "(CASE WHEN Title LIKE ? OR LearningContent LIKE ? OR ChallengesFaced LIKE ? OR RecommendationOrLessonLearned LIKE ? THEN 1 ELSE 0 END)"
        )
        score_params.extend([like, like, like, like])

    score_sql = " + ".join(score_exprs) if score_exprs else "0"

    if min_should and score_exprs:
        # Require at least min_should of the optional terms
        sql = f"""
            SELECT ID
            FROM dbo.Learnings
            WHERE {where_must}
              AND ({score_sql}) >= ?
        """
        all_params = params + score_params + [min_should]
    else:
        # No optional requirement
        sql = f"""
            SELECT ID
            FROM dbo.Learnings
            WHERE {where_must}
        """
        all_params = params

    ids: List[int] = []
    with conn.cursor() as cur:
        cur.execute(sql, all_params)
        for row in cur.fetchall():
            ids.append(int(row[0]))

    # Optional: dedupe & sort
    return sorted(set(ids))
