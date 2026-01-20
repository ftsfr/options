"""
Doit build file for SPX Options pipeline.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import chartbook

sys.path.insert(1, "./src/")

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"
OUTPUT_DIR = BASE_DIR / "_output"
NOTEBOOK_BUILD_DIR = OUTPUT_DIR / "_notebook_build"
OS_TYPE = "nix" if platform.system() != "Windows" else "windows"


def jupyter_execute_notebook(notebook):
    """Execute a notebook and convert to HTML."""
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "html",
            "--output-dir",
            str(OUTPUT_DIR),
            str(notebook),
        ],
        check=True,
    )


def jupyter_to_html(notebook):
    """Convert notebook to HTML without execution."""
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--output-dir",
            str(OUTPUT_DIR),
            str(notebook),
        ],
        check=True,
    )


def copy_notebook_to_build(notebook_path, dest_dir):
    """Copy notebook to build directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / notebook_path.name
    shutil.copy(notebook_path, dest_path)
    return dest_path


def task_config():
    """Create directories for data and output."""

    def create_dirs():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        NOTEBOOK_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "actions": [create_dirs],
        "targets": [DATA_DIR, OUTPUT_DIR, NOTEBOOK_BUILD_DIR],
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


def task_run_notebooks():
    """Execute summary notebook and convert to HTML."""
    notebook_py = BASE_DIR / "src" / "summary_options_ipynb.py"
    notebook_ipynb = NOTEBOOK_BUILD_DIR / "summary_options.ipynb"

    def run_notebook():
        # Convert py to ipynb
        subprocess.run(
            ["ipynb-py-convert", str(notebook_py), str(notebook_ipynb)],
            check=True,
        )
        # Execute the notebook
        jupyter_execute_notebook(notebook_ipynb)

    return {
        "actions": [run_notebook],
        "file_dep": [
            notebook_py,
            DATA_DIR / "ftsfr_cjs_option_returns.parquet",
            DATA_DIR / "ftsfr_hkm_option_returns.parquet",
        ],
        "targets": [
            notebook_ipynb,
            OUTPUT_DIR / "summary_options.html",
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
            NOTEBOOK_BUILD_DIR / "summary_options.ipynb",
        ],
        "targets": [BASE_DIR / "docs" / "index.html"],
        "verbosity": 2,
        "task_dep": ["run_notebooks"],
    }
