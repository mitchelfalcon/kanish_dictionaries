%%writefile pipeline_phase1.py
import pandas as pd
import json
import re

with open('grammar_logic.json', 'r', encoding='utf-8') as f:
    gema = json.load(f)

norm_rules = gema['data_ingestion']['logic_dna']['normalization']

def normalize(text):
    for k, v in norm_rules.items():
        text = text.replace(k, v)
    return text

def parse(text_id, line_uid, line_num, line_idx, text):
    tokens = []
    for seq, t in enumerate(text.split(), 1):
        clean = normalize(t.translate(str.maketrans('', '', '#?!*')))
        ttype = 'sign'
        if '{' in clean: 
            ttype = 'det'
        elif re.match(r'^\d', clean): 
            ttype = 'number'
        elif '|' in clean: 
            ttype = 'compound'
        
        tokens.append({
            'text_id': text_id,
            'line_uid': line_uid,
            'line_number': line_num,
            'line_index': line_idx,
            'token_seq': seq,
            'token_type': ttype,
            'raw_content': t,
            'unicode_val': clean,
            'is_damaged': '#' in t,
            'is_uncertain': '?' in t,
            'is_corrected': '!' in t,
            'is_collated': '*' in t
        })
    return tokens

df = pd.read_csv('test.csv')
res = []

for i, r in df.iterrows():
    res.extend(parse(
        r['text_id'], 
        f"{r['text_id']}.{r['line_start']}", 
        f"{r['line_start']}.", 
        i, 
        str(r['transliteration'])
    ))

with open('phase1_tokens.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print(json.dumps(res[:5], ensure_ascii=False, indent=2))
