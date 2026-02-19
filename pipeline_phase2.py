%%writefile pipeline_phase2.py
import json
import re

with open('phase1_tokens.json', 'r', encoding='utf-8') as f:
    tokens = json.load(f)

with open('atf-flat-tokens.json', 'r', encoding='utf-8') as f:
    logic_dna = json.load(f)['atf_logic_dna']

def apply_logic_dna(token):
    val = token['unicode_val']
    
    if token['token_type'] == 'sign':
        if val.isupper():
            token['reading_type'] = 'logogram'
        elif val.islower():
            token['reading_type'] = 'syllable'
        else:
            token['reading_type'] = 'mixed'
    else:
        token['reading_type'] = None
        
    token['modifiers'] = []
    for mod in logic_dna['modifiers']['allowed']:
        mod_str = f"@{mod}"
        if mod_str in val:
            token['modifiers'].append(logic_dna['modifiers']['definitions'][mod_str])
            val = val.replace(mod_str, "")
            
    token['is_uncertain_reading'] = '$' in val
    token['is_logogram_marker'] = '~' in val
    
    token['clean_value'] = re.sub(r'[\$~]', '', val).split('@')[0]
    
    return token

normalized = [apply_logic_dna(t) for t in tokens]

with open('phase2_normalized.json', 'w', encoding='utf-8') as f:
    json.dump(normalized, f, ensure_ascii=False, indent=2)

print(json.dumps(normalized[:5], ensure_ascii=False, indent=2))
