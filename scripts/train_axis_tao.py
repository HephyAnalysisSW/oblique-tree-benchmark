"""Isolated axis experiment using the existing resumable training state machine."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tao.axis import AxisTAOClassifier,assert_axis
from scripts import train_tao_fully as worker


def main():
    config=sys.argv[sys.argv.index('--config')+1]
    plan=json.loads((ROOT/config).read_text())
    if plan['split_search']!='axis_threshold' or plan['split_alpha']!=0 or plan['initialization']!='cart':
        raise ValueError('Expected the isolated CART-initialized axis-aligned scan policy')
    for path,digest in plan['implementation_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Training implementation changed; version the experiment before continuing: '+path)
    worker.HardTAOClassifier=AxisTAOClassifier
    original=worker.save
    def checked_save(run,model,*args):
        assert_axis(model)
        return original(run,model,*args)
    worker.save=checked_save
    return worker.main()

if __name__=='__main__':sys.exit(main())
