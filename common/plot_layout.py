"""Readable public figure paths; training artifacts keep their original paths."""
from pathlib import Path
import yaml


def publication_path(source: Path, root: Path) -> Path:
    try:
        relative = source.relative_to(root)
    except ValueError:
        return Path(source.name)
    if source.suffix.lower() not in {'.png', '.pdf', '.svg', '.gif'}:
        return relative
    for config_path in sorted((root / 'configs').glob('*.yaml')):
        config = yaml.safe_load(config_path.read_text())
        run = config.get('run_name', '')
        if not run or not source.stem.startswith(run + '_'):
            continue
        model = config.get('model', {})
        if 'tao' in model:
            tao = model['tao']
            layout = Path(f"tao_{tao['trees']}tree_depth{tao['depth']}_{tao['leaf_mode']}")
        elif 'xgboost' in model:
            bdt = model['xgboost']
            # Multiclass XGBoost builds one tree per class per boosting round.
            layout = Path(f"xgboost_{bdt['n_estimators']}rounds_depth{bdt['max_depth']}")
        else:
            continue
        return Path('plots') / layout / source.name
    if source.stem.endswith('_reference_vs_retrained_roc'):
        return Path('plots/comparisons') / source.name
    if relative.parent == Path('results/tao_steps'):
        group = 'comparisons' if source.stem.startswith('pilot_') else 'diagnostics'
        return Path('plots') / group / source.name
    return relative
