"""Benchmark output destinations; environment variables can override these."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
plot_directory = os.environ.get("TREE_BENCHMARK_PLOT_DIRECTORY", "/groups/hephy/cms/robert.schoefbeck/www/tree_benchmark")
model_directory = str(ROOT / "models")
