%%writefile pipeline_phase9.py
import json
import pandas as pd

with open('site-periods.json', 'r', encoding='utf-8') as f:
    periods = json.load(f)

df_periods = pd.DataFrame(periods)
period_summary = df_periods.groupby('period')['count'].sum().reset_index()

with open('schema-metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

sub_df = pd.read_csv('final_submission.csv')
sub_df['period_inferred'] = 'Old Assyrian'
sub_df['project_scope'] = metadata.get('meta_header', {}).get('project_scope', 'Kanish_Vortex')

# Drop unused columns for a strict standard Kaggle submission layout
columns_for_kaggle = ['id', 'transliteration', 'lemma', 'translation']
final_kaggle_df = sub_df[columns_for_kaggle]

final_kaggle_df.to_csv('submission_certified_final.csv', index=False, encoding='utf-8')

print(final_kaggle_df.head(4).to_json(orient='records', indent=2, force_ascii=False))
