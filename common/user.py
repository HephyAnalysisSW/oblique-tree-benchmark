"""Benchmark output destinations; environment variables can override these."""
import os
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
plot_directory = os.environ.get("TREE_BENCHMARK_PLOT_DIRECTORY", "/groups/hephy/cms/robert.schoefbeck/www/tree_benchmark")
if "TREE_BENCHMARK_PLOT_DIRECTORY" not in os.environ:
    try:
        if shutil.disk_usage('/groups').free < 50*1024**2:
            plot_directory = str(ROOT / 'results/publish_cache/www/tree_benchmark')
    except OSError:
        plot_directory = str(ROOT / 'results/publish_cache/www/tree_benchmark')
model_directory = str(ROOT / "models")
