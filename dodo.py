"""
Doit build file for SPX Options pipeline.
"""

import os
import platform
import sys
from pathlib import Path

import chartbook

sys.path.insert(1, "./src/")

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"
OUTPUT_DIR = BASE_DIR / "_output"
OS_TYPE = "nix" if platform.system() != "Windows" else "windows"



## Helpers for handling Jupyter Notebook tasks
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


# fmt: off
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def task_config():
    """Create directories for data and output."""
    def create_dirs():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "actions": [create_dirs],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "verbosity": 2,
    }


def task_pull():
    """Pull SPX options data from WRDS OptionMetrics."""
    return {
        "actions": ["python src/pull_option_data.py"],
        "file_dep": ["src/pull_option_data.py"],
        "targets": [
            DATA_DIR / "data_1996-01_2012-01.parquet",
            DATA_DIR / "data_2012-02_2019-12.parquet",
            DATA_DIR / "data_1996-01_2019-12.parquet",
        ],
        "verbosity": 2,
        "task_dep": ["config"],
    }


def task_calc_filters():
    """Apply CJS filters to raw options data."""
    return {
        "actions": ["python src/calc_filters.py"],
        "file_dep": [
            "src/calc_filters.py",
            "src/level_1_filters.py",
            "src/level_2_filters.py",
            "src/level_3_filters.py",
            DATA_DIR / "data_1996-01_2019-12.parquet",
        ],
        "targets": [
            DATA_DIR / "L1_filtered_1996-01_2019-12.parquet",
            DATA_DIR / "L2_filtered_1996-01_2019-12.parquet",
            DATA_DIR / "L3_filtered_1996-01_2019-12.parquet",
            DATA_DIR / "spx_filtered_final_1996-01_2019-12.parquet",
        ],
        "verbosity": 2,
        "task_dep": ["pull"],
    }


def task_calc_portfolios():
    """Build CJS and HKM portfolios from filtered data."""
    return {
        "actions": ["python src/calc_portfolios.py"],
        "file_dep": [
            "src/calc_portfolios.py",
            DATA_DIR / "spx_filtered_final_1996-01_2019-12.parquet",
        ],
        "targets": [
            DATA_DIR / "cjs_portfolio_returns_1996-01_2019-12.parquet",
            DATA_DIR / "hkm_portfolio_returns_1996-01_2019-12.parquet",
        ],
        "verbosity": 2,
        "task_dep": ["calc_filters"],
    }


def task_format():
    """Create FTSFR standardized datasets."""
    return {
        "actions": ["python src/create_ftsfr_datasets.py"],
        "file_dep": [
            "src/create_ftsfr_datasets.py",
            DATA_DIR / "cjs_portfolio_returns_1996-01_2019-12.parquet",
            DATA_DIR / "hkm_portfolio_returns_1996-01_2019-12.parquet",
        ],
        "targets": [
            DATA_DIR / "ftsfr_cjs_option_returns.parquet",
            DATA_DIR / "ftsfr_hkm_option_returns.parquet",
        ],
        "verbosity": 2,
        "task_dep": ["calc_portfolios"],
    }


notebook_tasks = {
    "summary_options_ipynb": {
        "path": "./src/summary_options_ipynb.py",
        "file_dep": [
            DATA_DIR / "ftsfr_cjs_option_returns.parquet",
            DATA_DIR / "ftsfr_hkm_option_returns.parquet",
        ],
        "targets": [],
    },
}
notebook_files = []
for notebook in notebook_tasks.keys():
    pyfile_path = Path(notebook_tasks[notebook]["path"])
    notebook_files.append(pyfile_path)


def task_run_notebooks():
    """Execute summary notebook and convert to HTML."""
    for notebook in notebook_tasks.keys():
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix(".ipynb")
        yield {
            "name": notebook,
            "actions": [
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
            "task_dep": ["format"],
        }


def task_generate_charts():
    """Generate interactive HTML charts."""
    return {
        "actions": ["python src/generate_chart.py"],
        "file_dep": [
            "src/generate_chart.py",
            DATA_DIR / "ftsfr_hkm_option_returns.parquet",
        ],
        "targets": [
            OUTPUT_DIR / "options_returns_replication.html",
            OUTPUT_DIR / "options_cumulative_returns.html",
        ],
        "verbosity": 2,
        "task_dep": ["format"],
    }


def task_generate_pipeline_site():
    """Generate chartbook documentation site."""
    return {
        "actions": ["chartbook build -f"],
        "file_dep": [
            "chartbook.toml",
            *notebook_files,
            OUTPUT_DIR / "options_returns_replication.html",
            OUTPUT_DIR / "options_cumulative_returns.html",
        ],
        "targets": [BASE_DIR / "docs" / "index.html"],
        "verbosity": 2,
        "task_dep": ["run_notebooks", "generate_charts"],
    }
