%%writefile pipeline_phase5.py
import json

with open('query_builder_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)['query_builder_config']

table = config['target_table']

def translate_query(query):
    if '&&' in query:
        parts = [p.strip() for p in query.split('&&')]
        sql = f"SELECT t1.line_uid FROM {table} t1 JOIN {table} t2 ON t1.line_uid = t2.line_uid WHERE t1.clean_value = '{parts[0]}' AND t2.clean_value = '{parts[1]}';"
        return sql
    elif '||' in query:
        parts = [p.strip() for p in query.split('||')]
        sql = f"SELECT line_uid FROM {table} WHERE clean_value = '{parts[0]}' UNION SELECT line_uid FROM {table} WHERE clean_value = '{parts[1]}';"
        return sql
    elif '!' in query:
        parts = [p.strip() for p in query.split('!')]
        sql = f"SELECT line_uid FROM {table} WHERE clean_value = '{parts[0]}' EXCEPT SELECT line_uid FROM {table} WHERE clean_value = '{parts[1]}';"
        return sql
    elif '->' in query:
        parts = [p.strip() for p in query.split('->')]
        a = parts[0]
        dist = int(parts[1].split()[0])
        b = parts[2]
        sql = f"SELECT t1.line_uid FROM {table} t1 JOIN {table} t2 ON t1.text_id = t2.text_id AND t2.token_seq = t1.token_seq + {dist} WHERE t1.clean_value = '{a}' AND t2.clean_value = '{b}';"
        return sql
    return f"SELECT * FROM {table} WHERE clean_value = '{query}';"

queries = [
    "szu && ti",
    "szu || ba",
    "szu ! ti",
    "szu -> 1 word -> ba"
]

results = {q: translate_query(q) for q in queries}

with open('phase5_queries.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
