"""Build scientific table/plot from the frozen, checked-in result snapshot.

No training environment or datasets are needed. --plot additionally needs matplotlib.
The existing implementation-oriented assets and provenance are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def metric(record):
    return record['metrics']['signal_efficiency_at_fixed_fpr']

def eligible(row):
    return [r for stage, r in row['stages'].items()
            if stage.startswith('leaf_only') or
            (row['phase'] == 'complete' and stage.startswith('full'))]

def number(value):
    return format(value, '.8g')

def expression(weights, bias):
    terms = [(v, rf'x_{{{i+1}}}') for i, v in enumerate(weights) if v != 0]
    if bias != 0:
        terms.append((bias, ''))
    parts = []
    for value, variable in terms:
        n = number(abs(value))
        if 'e' in n:
            mantissa, exponent = n.split('e')
            n = mantissa + r'\cdot10^{' + str(int(exponent)) + '}'
        term = ('' if n == '1' and variable else n) + variable
        parts.append(('-' if value < 0 else '') + term if not parts else
                     (' - ' if value < 0 else ' + ') + term)
    return parts or ['0']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    snapshot = json.loads((ROOT/'sources/results_snapshot.json').read_text())
    spec = json.loads((ROOT/'sources/compact_standardized_model.json').read_text())
    rows = []
    for row in snapshot['models']:
        stages = row['stages']
        penalties = row['selection_penalties']
        pair = f"{penalties['split_alpha']:g}/{penalties['leaf_alpha']:g}".replace('e-0', 'e-')
        star = r'$^*$' if row['case']['trees'] == 1 else ''
        state = {'complete': 'complete', 'inconclusive': 'timeout', 'separated': 'separated'}[row['phase']]
        scores = [metric(stages['parent']), metric(stages['leaf_only']), max(map(metric, eligible(row)))]
        rows.append(pair + star + ' & ' + str(row['case']['parent_nonzeros']) +
                    ' & ' + ' & '.join(f'{100*v:.2f}' for v in scores) + ' & ' + state + r' \\')
    (ROOT/'generated/research_sfr_results.tex').write_text('\n'.join(rows) + '\n' + r'\bottomrule' + '\n')
    equations = []
    for t, node, path, label in [(0, 0, r'\varnothing', 'root'), (0, 1, 'L', 'constant'), (0, 2, 'R', 'oblique')]:
        tree = spec['trees'][t]
        terms = expression(tree['split_weights'][node], tree['split_bias'][node])
        relation = '=' if not any(tree['split_weights'][node]) and tree['split_bias'][node] == 0 else r'\simeq'
        chunks = [''.join(terms[i:i+3]) for i in range(0, len(terms), 3)]
        equations.append(r'\begin{equation}\begin{aligned}' + '\n' +
                         rf'h_{{{t+1},{path}}}(x)&{relation} ' + chunks[0] +
                         ''.join('\\\\\n&\\quad' + chunk for chunk in chunks[1:]) + '\n' +
                         r'\end{aligned}\label{eq:example-' + label + r'}\end{equation}')
    (ROOT/'generated/research_cuts.tex').write_text('\n'.join(equations) + '\n')
    tree = spec['trees'][1]
    tiny = expression(tree['split_weights'][1], tree['split_bias'][1])
    (ROOT/'generated/research_tiny_cut.tex').write_text(
        r'\begin{equation}h_{2,L}(x)\simeq ' + ''.join(tiny) +
        r'\label{eq:example-tiny}\end{equation}' + '\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        labels = {'split300': 'Compact refit', 'both10': '520 coefficients',
                  'leaf3': 'Higher efficiency', 'both100': '317 coefficients'}
        for row in snapshot['models']:
            n = row['case']['parent_nonzeros']
            parent = 100*metric(row['stages']['parent'])
            best = 100*max(map(metric, eligible(row)))
            ax.plot([n, n], [parent, best], color='#999999', lw=.8)
            ax.scatter(n, parent, marker='o', facecolors='none', edgecolors='#999999', s=22)
            ax.scatter(n, best, color='#2166ac', s=22)
            if row['case']['id'] in labels:
                ax.annotate(labels[row['case']['id']], (n, best), xytext=(4, 5),
                            textcoords='offset points', fontsize=8)
        ax.scatter([], [], marker='o', facecolors='none', edgecolors='#999999', label='L1-selected parent')
        ax.scatter([], [], color='#2166ac', label='Best eligible refit')
        ax.axhline(100*snapshot['xgboost_metrics']['signal_efficiency_at_fixed_fpr'],
                   color='#222222', ls='--', lw=1, label='XGBoost')
        ax.set(xlabel='Frozen support budget (including biases)',
               ylabel='Signal efficiency at 1% background [%]', ylim=(50.5, 58), xlim=(280, 1020))
        ax.grid(alpha=.2)
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        fig.savefig(ROOT/'figures/sfr_research.pdf')
        plt.close(fig)
    paths = ['sources/results_snapshot.json', 'sources/compact_standardized_model.json',
             'generated/research_sfr_results.tex', 'generated/research_cuts.tex',
             'generated/research_tiny_cut.tex']
    if (ROOT/'figures/sfr_research.pdf').exists():
        paths.append('figures/sfr_research.pdf')
    provenance = {'inputs_and_outputs': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                  'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'note': 'Derived only from frozen checked-in snapshots; no new model evaluation.'}
    (ROOT/'sources/scientific_assets.json').write_text(json.dumps(provenance, indent=2) + '\n')

if __name__ == '__main__':
    main()
