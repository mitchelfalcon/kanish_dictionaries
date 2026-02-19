%%writefile pipeline_phase4.py
import json
import pandas as pd
import sqlite3

with open('phase3_parsed.json', 'r', encoding='utf-8') as f:
    tokens = json.load(f)

df = pd.DataFrame(tokens)

cols = [
    'text_id', 'line_uid', 'line_number', 'line_index', 'token_seq', 
    'token_type', 'raw_content', 'unicode_val', 'clean_value', 
    'is_damaged', 'is_uncertain', 'is_corrected', 'is_collated', 
    'reading_type', 'pos_tag', 'case'
]

existing_cols = [c for c in cols if c in df.columns]
df_clean = df[existing_cols].copy()

df_clean.to_csv('atf_flat_db.csv', index=False, encoding='utf-8')

conn = sqlite3.connect('atf_flat_db.sqlite')
df_clean.to_sql('atf_flat_tokens', conn, if_exists='replace', index=False)
conn.close()

print(df_clean.head(5).to_json(orient='records', indent=2, force_ascii=False))
