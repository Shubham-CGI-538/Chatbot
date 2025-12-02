# backfill_embeddings.py (local-only)
import os, json, pyodbc
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

CONN_STR = os.getenv("DB_CONN_STR")
# Point to the local directory you copied
EMBED_MODEL_DIR = os.getenv("EMBED_MODEL_DIR", r"C:\Users\shubham.bansal\Projects\chatbot\Chatbot\models\all-MiniLM-L6-v2")
BATCH = int(os.getenv("EMBED_BATCH", "200"))

# Important: pass the directory path, not the model name
model = SentenceTransformer(EMBED_MODEL_DIR)  

def combine_text(row):
    parts = [
        row.Title or "",
        row.LearningContent or "",
        row.ChallengesFaced or "",
        row.RecommendationOrLessonLearned or ""
    ]
    return "\n".join(parts).strip()

def main():
    conn = pyodbc.connect(CONN_STR)
    cur = conn.cursor()
    cur.fast_executemany = True

    rows = []
    cur_sel = conn.cursor()
    cur_sel.execute("""
        SELECT l.ID, l.Title, l.LearningContent, l.ChallengesFaced, l.RecommendationOrLessonLearned
        FROM dbo.Learnings l
        LEFT JOIN dbo.LearningsEmbeddings e ON e.ID = l.ID
        WHERE e.ID IS NULL
    """)
    colnames = [d[0] for d in cur_sel.description]
    for r in cur_sel.fetchall():
        obj = type("Row", (), dict(zip(colnames, r)))
        rows.append(obj)

    total = len(rows)
    print(f"Found {total} rows requiring embeddings.")

    for i in range(0, total, BATCH):
        batch = rows[i:i+BATCH]
        texts = [combine_text(r) for r in batch]
        if not texts:
            continue

        # Local inference, fully offline
        embeddings = model.encode(
            texts,
            batch_size=min(64, BATCH),
            show_progress_bar=False,
            normalize_embeddings=False
        )

        data = [(batch[j].ID, json.dumps(embeddings[j].tolist())) for j in range(len(batch))]
        cur.executemany("""
            MERGE dbo.LearningsEmbeddings AS t
            USING (SELECT ? AS ID, ? AS EmbeddingJson) AS s
            ON t.ID = s.ID
            WHEN MATCHED THEN UPDATE SET EmbeddingJson = s.EmbeddingJson, LastUpdatedUtc = SYSUTCDATETIME()
            WHEN NOT MATCHED THEN INSERT (ID, EmbeddingJson) VALUES (s.ID, s.EmbeddingJson);
        """, data)
        conn.commit()
        print(f"Embedded {i + len(batch)} / {total}")

    cur_sel.close()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()