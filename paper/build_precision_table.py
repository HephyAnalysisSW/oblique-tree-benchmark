"""Render the paper table from the checked-in, completed conifer validation."""
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent


def main():
    report = json.loads((PAPER.parent / 'docs/conifer_xgboost_v1/validation.json').read_text())
    assert report['complete'] and not report['test_evaluated']
    rows = []
    for profile in report['profiles']:
        if not profile['name'].startswith('fixed_'):
            continue
        assert profile['hls_cpp_emulation']['error_against_cmssw']['exact_mismatches'] == 0
        widths = [int(re.match(r'ap_fixed<(\d+),', profile[key]).group(1)) for key in ('input', 'score')]
        metric = profile['metrics']
        assert metric['accepted_background'] <= metric['background_budget']
        rows.append(f"{widths[0]} & {widths[1]} & "
                    f"{100 * metric['signal_efficiency_at_1pct_background']:.2f} & "
                    f"{100 * metric['strict_signal_efficiency']:.2f} & {metric['auc']:.5f} " + r'\\')
    assert len(rows) == 6
    (PAPER / 'generated/xgboost_precision_results.tex').write_text('\n'.join(rows) + '\n' + r'\bottomrule' + '\n')


if __name__ == '__main__':
    main()
