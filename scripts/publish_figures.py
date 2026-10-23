"""Publish existing figures using the same layout as future automatic uploads."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import syncer, user
from common.plot_layout import publication_path

for path in sorted((user.ROOT / 'results').rglob('*.png')):
    syncer.register_file(path)
    print(publication_path(path, user.ROOT), flush=True)
