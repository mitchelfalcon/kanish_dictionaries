%%writefile pipeline_phase6.py
import json
import sqlite3
import pandas as pd

with open('phase5_queries.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

conn = sqlite3.connect('atf_flat_db.sqlite')
results = {}

for q_name, sql in queries.items():
    try:
        df = pd.read_sql_query(sql, conn)
        results[q_name] = df.to_dict(orient='records')
    except Exception as e:
        results[q_name] = {"error": str(e)}

conn.close()

with open('phase6_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

summary = {k: v[:5] if isinstance(v, list) else v for k, v in results.items()}
print(json.dumps(summary, indent=2, ensure_ascii=False))
