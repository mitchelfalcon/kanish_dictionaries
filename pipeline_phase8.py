%%writefile pipeline_phase8.py
import json
import pandas as pd

with open('phase7_dashboard_export.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('roles-schema.json', 'r', encoding='utf-8') as f:
    roles = json.load(f)

role_map = {r['term_akkadian']: r['translation_en'] for r in roles}

final_sub = []
for row in data:
    translation_parts = []
    for lemma in row['lemma_gematria_encoded'].split():
        clean_term = lemma.split('[')[0]
        if clean_term in role_map:
            translation_parts.append(role_map[clean_term])
        else:
            translation_parts.append(clean_term)
            
    final_sub.append({
        "id": row['line_uid'],
        "transliteration": row['transliteration'],
        "lemma": row['lemma_gematria_encoded'],
        "translation": " ".join(translation_parts)
    })

df = pd.DataFrame(final_sub)
df.to_csv('final_submission.csv', index=False, encoding='utf-8')

print(df.head(5).to_json(orient='records', indent=2, force_ascii=False))
