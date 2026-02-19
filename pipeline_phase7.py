%%writefile pipeline_phase7.py
import json
import sqlite3
import pandas as pd

with open('economic-schema.json', 'r', encoding='utf-8') as f:
    econ_schema = json.load(f)

gematria_lookup = {item['term_akkadian']: item.get('gematria_data', {}) for item in econ_schema}

conn = sqlite3.connect('atf_flat_db.sqlite')
df = pd.read_sql_query("SELECT * FROM atf_flat_tokens ORDER BY text_id, line_index, token_seq", conn)
conn.close()

submission = []
for line_uid, group in df.groupby('line_uid', sort=False):
    lemmas = []
    for _, row in group.iterrows():
        clean_val = str(row['clean_value'])
        pos = str(row['pos_tag']) if pd.notna(row['pos_tag']) else 'X'
        
        gematria_hash = gematria_lookup.get(clean_val, {}).get('hash', '0x00000000')
        lemmas.append(f"{clean_val}[{gematria_hash}]{pos}")
        
    submission.append({
        "line_uid": line_uid,
        "transliteration": " ".join(group['raw_content'].astype(str)),
        "lemma_gematria_encoded": " ".join(lemmas)
    })

sub_df = pd.DataFrame(submission)
sub_df.to_csv('phase7_dashboard_export.csv', index=False, encoding='utf-8')

with open('phase7_dashboard_export.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=2)

print(json.dumps(submission[:5], ensure_ascii=False, indent=2))
