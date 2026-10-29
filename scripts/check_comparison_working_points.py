"""Check evaluator-specific strict cuts on the complete fixed validation split.

Requires the archived common C++ score files (or a reproduced benchmark).
No test arrays are read; model coefficients and working points are not changed.
"""
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from tao.portable import predict_margin


def main():
    manifest=json.loads((ROOT/'fpga_handoff/comparison_v1/manifest.json').read_text())
    dataset=ROOT/'data/qg_vs_wzt_standardized.npz'
    assert hashlib.sha256(dataset.read_bytes()).hexdigest()==manifest['dataset_sha256']
    with np.load(dataset,allow_pickle=False) as data:x,y=data['X_val'].astype(float),data['y_val']
    rows=[]
    for c in manifest['candidates']:
        score=np.fromfile(ROOT/('artifacts/ten_model_comparison_v1/'+c['id']+'.bin'),dtype='<f8');assert len(score)==len(y)
        threshold=c['metrics']['strict_threshold'];cpp=int(np.count_nonzero(score[y==0]>threshold))
        assert cpp==c['metrics']['accepted_background'] and cpp<=525
        row=dict(id=c['id'],cpp_threshold=threshold,cpp_background_events=cpp)
        if c['kind']=='tao':
            spec=json.loads((ROOT/c['model']).read_text());margin,_,_=predict_margin(spec,x)
            wp=spec['working_points']['portable_numpy'];numpy_count=int(np.count_nonzero(margin[y==0]>wp['signal_margin_threshold']))
            assert numpy_count==wp['false_positives'] and numpy_count<=525
            row.update(numpy_threshold=wp['signal_margin_threshold'],numpy_background_events=numpy_count,
                numpy_events_using_cpp_threshold=int(np.count_nonzero(margin[y==0]>threshold)))
        rows.append(row)
    report=dict(all_passed=True,dataset_sha256=manifest['dataset_sha256'],events=len(y),test_evaluated=False,checks=rows,
        note='Thresholds are evaluator-specific, even for binary64; Compact demonstrates an ulp-sensitive boundary. Recheck after arithmetic changes.')
    out=ROOT/'docs/ten_model_comparison_v1/working_point_checks.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
