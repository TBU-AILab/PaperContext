from pathlib import Path
import pandas as pd
import json
import numpy as np

# path to CSV (adjust if needed)
csv_path = Path(r"\\10.5.8.121\WorkBench\PaperContext\tasks_sorted") / "solutions_summary_enriched_with_best.csv"

df = pd.read_csv(csv_path)

# 🔥 CRITICAL FIX: remove NaN for JSON
df = df.replace({np.nan: None})

attribute_columns = [
    "actual_iteration", "metaheuristic", "metaheuristic_family",
    "local_search", "adaptation", "initialization", "restart",
    "surrogate", "elitism", "archive", "niching_or_diversity",
    "hybridized", "population_based", "stochastic",
    "llm_family", "context", "best_mean", "best_min"
]

sort_columns = ["Mean", "Min", "Max", "Median", "STD", "Mean_rescaled", "Min_rescaled"]

metadata = {
    "attribute_columns": attribute_columns,
    "sort_columns": sort_columns,
    "default_visible_attributes": attribute_columns,
    "default_sort_column": "Mean",
    "default_sort_direction": "desc",
    "text_search_columns": [
        "task_id", "solution_dir", "py_file",
        "short_rationale", "metaheuristic_family",
        "llm_family", "context"
    ],
    "attribute_values": {
        col: sorted(df[col].dropna().unique().tolist())
        for col in attribute_columns if col in df.columns
    }
}

records = df.to_dict(orient="records")

payload = {
    "metadata": metadata,
    "records": records
}

# 🔥 SAVE HERE (IMPORTANT)
output_path = Path(__file__).parent / "plotly_task_feature_map" / "dataiter.json"
output_path.parent.mkdir(exist_ok=True)

with open(output_path, "w") as f:
    json.dump(payload, f, indent=2)

print("Saved to:", output_path)