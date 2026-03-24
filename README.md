# Repository with support data

This repository contains experimental artifacts and analysis tools for studying the impact of LLM conversational context on algorithm generation performance.

### Structure Overview

The repository is centered around experimental evaluation of LLM-generated algorithmic solutions under different context configurations.

- **Data preparation**
  - `prepare_data.py` – Transforms raw experimental outputs into structured datasets.
  - `cleanup.py` – Performs post-processing and filtering of results.

- **Data artifacts**
  - `data.json` – Intermediate or aggregated dataset used for analysis.
  - `solutions_summary_enriched.csv` – Tabular summary of solutions enriched with performance metrics and features.

- **Visualization**
  - `index.html` – Interactive visualization interface.
  - `plotly_task_feature_map/` – Data used for feature-space visualization of tasks and solutions.

- **Experimental results (`tasks_sorted/`)**
  - Organized hierarchically by:
    1. **Model family** (e.g., `gemini`)
    2. **Context strategy** (e.g., `all_`, `last_`, `*_best`, `*_3best`)
    3. **Task instance (`t_<UUID>`)**

- **Task-level structure**
  Each task directory contains:
  - `messages.csv` – Full interaction history with the LLM.
  - `solution/` – Multiple generated candidate solutions.
  - `stat/` – Statistical evaluation and benchmarking results.
  - `task.pkl` – Serialized representation of the task (including configuration and state).

This structure enables reproducible analysis of how different LLM context strategies affect the quality and behavior of generated solutions.