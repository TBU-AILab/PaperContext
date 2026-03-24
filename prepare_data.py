import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

CSV_PATH = Path(r"Z:\PaperContext\tasks_sorted\solutions_summary_enriched_with_best.csv")
OUTPUT_DIR = Path("plotly_task_feature_map")

ATTRIBUTE_COLS = [
    "actual_iteration",
    "metaheuristic",
    "metaheuristic_family",
    "local_search",
    "adaptation",
    "initialization",
    "restart",
    "surrogate",
    "elitism",
    "archive",
    "niching_or_diversity",
    "hybridized",
    "population_based",
    "stochastic",
    "llm_family",
    "context",
    "best_mean",
    "best_min",
]

SORT_COLS = ["Mean", "Min", "Max", "Median", "STD", "Mean_rescaled", "Min_rescaled"]
TEXT_COLS = ["task_id", "solution_dir", "py_file", "short_rationale", "metaheuristic_family", "llm_family", "context"]


def normalize_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    return str(value)


def build_data_json(csv_path: Path = CSV_PATH, output_dir: Path = OUTPUT_DIR) -> Path:
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in df.iterrows():
        record = {col: normalize_value(row[col]) for col in df.columns}
        record["_search"] = " | ".join(
            str(record.get(col, "")) for col in TEXT_COLS if record.get(col) is not None
        ).lower()
        records.append(record)

    metadata = {
        "attribute_columns": ATTRIBUTE_COLS,
        "sort_columns": SORT_COLS,
        "default_visible_attributes": ATTRIBUTE_COLS,
        "default_sort_column": "Mean",
        "default_sort_direction": "desc",
        "text_search_columns": TEXT_COLS,
        "attribute_values": {},
    }

    for col in ATTRIBUTE_COLS:
        values = pd.Series(df[col]).dropna().unique().tolist()
        values = sorted(values, key=lambda x: str(x))
        metadata["attribute_values"][col] = [normalize_value(v) for v in values]

    output_path = output_dir / "data.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "records": records}, f, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    out = build_data_json()
    print(f"Created: {out.resolve()}")
