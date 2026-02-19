%%writefile pipeline_phase10.py
import json
import pandas as pd

with open('kanish_geodata_master.json', 'r', encoding='utf-8') as f:
    geo = json.load(f)

df = pd.read_csv('submission_certified_final.csv')

dashboard_data = {
    "system_status": "DETERMINISTIC_MODE_ENABLED",
    "total_processed_lines": len(df),
    "geospatial_anchors": geo.get("realms_spatial_anchor", {}),
    "route_metrics": geo.get("route_weights", {})
}

with open('phase10_dashboard_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

print(json.dumps(dashboard_data, ensure_ascii=False, indent=2))
