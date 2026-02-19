%%writefile pipeline_phase3.py
import json

with open('phase2_normalized.json', 'r', encoding='utf-8') as f:
    tokens = json.load(f)

with open('grammar_logic.json', 'r', encoding='utf-8') as f:
    grammar = json.load(f)['morphology_engine']

def parse_morphology(token):
    if token['token_type'] == 'sign':
        token['morph_template'] = grammar.get('noun_phrase_template', [])
        token['pos_tag'] = 'UNKNOWN'
        
        val = token.get('clean_value', '')
        if val.endswith('um'):
            token['pos_tag'] = 'N'
            token['case'] = 'nominative'
        elif val.endswith('am'):
            token['pos_tag'] = 'N'
            token['case'] = 'accusative'
        elif val.endswith('im'):
            token['pos_tag'] = 'N'
            token['case'] = 'genitive'
        else:
            token['case'] = 'oblique/construct/other'
    else:
        token['morph_template'] = None
        token['pos_tag'] = None
        token['case'] = None
        
    return token

parsed_tokens = [parse_morphology(t) for t in tokens]

with open('phase3_parsed.json', 'w', encoding='utf-8') as f:
    json.dump(parsed_tokens, f, ensure_ascii=False, indent=2)

print(json.dumps(parsed_tokens[:5], ensure_ascii=False, indent=2))
