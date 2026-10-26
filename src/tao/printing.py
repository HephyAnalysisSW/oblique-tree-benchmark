"""Read-only equations for the binary portable tree interface.

Kept separate from the pinned training implementation: inspecting a model does
not change its parameters, support masks, or training checkpoint identities.
"""
import copy
import math
import re
import sys


class TreePrinter:
    """Print every affine split/leaf, including algebraically unreachable nodes.

    Seventeen significant digits round-trip binary64 coefficients. Smaller
    precision is a display choice only, never pruning or an inference export.
    Feature names denote the input space in the supplied portable model JSON.
    """
    def __init__(self, specification, precision=17):
        if isinstance(precision, bool) or not isinstance(precision, int) or not 1 <= precision <= 17:
            raise ValueError('Precision must be an integer from 1 to 17')
        self.spec = copy.deepcopy(specification)
        self.precision = precision
        s = self.spec
        if s.get('schema') != 'tree-benchmark.binary-hard-oblique.v1':
            raise ValueError('Expected the binary portable tree schema')
        self.features = s['features']
        if len(self.features) != s['feature_count'] or len(set(self.features)) != len(self.features):
            raise ValueError('Feature names must be unique and match the input width')
        if not all(isinstance(f, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', f) for f in self.features):
            raise ValueError('Feature names must be printable identifiers')
        if s['input_space'] not in ('raw', 'standardized') or s['leaf_mode'] not in ('linear', 'constant'):
            raise ValueError('Unknown input space or leaf mode')
        if s['depth'] < 0 or s['internal_count'] != 2**s['depth']-1 or s['leaf_count'] != 2**s['depth']:
            raise ValueError('Inconsistent binary-tree topology')
        if len(s['trees']) != s['tree_count'] or len(s['classes']) != 2:
            raise ValueError('Inconsistent tree/class count')
        for tree in s['trees']:
            if len(tree['split_weights']) != s['internal_count'] or len(tree['split_bias']) != s['internal_count']:
                raise ValueError('Inconsistent split arrays')
            if len(tree['leaf_parameters']) != s['leaf_count']:
                raise ValueError('Inconsistent leaf array')
            groups = [(tree['split_weights'], s['feature_count']),
                      (tree['leaf_parameters'], s['feature_count']+1 if s['leaf_mode']=='linear' else 1)]
            for rows, width in groups:
                if any(len(row) != width for row in rows): raise ValueError('Inconsistent coefficient width')
                if any(not math.isfinite(float(v)) for row in rows for v in row): raise ValueError('Nonfinite coefficient')
            if any(not math.isfinite(float(v)) for v in tree['split_bias']): raise ValueError('Nonfinite bias')

    def affine(self, weights, bias):
        """Return a signed expression; omit only coefficients exactly equal to zero."""
        if len(weights) != len(self.features): raise ValueError('Wrong affine input width')
        terms = [(float(w), name) for w, name in zip(weights, self.features) if w != 0]
        if bias != 0: terms.append((float(bias), None))
        parts = []
        for value, name in terms:
            number = format(abs(value), f'.{self.precision}g')
            term = number if name is None else number + ' * ' + name
            parts.append(('-' if value < 0 else '') + term if not parts else (' - ' if value < 0 else ' + ') + term)
        return ''.join(parts) or '0'

    def records(self, tree_index=0):
        """Preorder records with exact coefficients and structural reachability."""
        if isinstance(tree_index, bool) or not isinstance(tree_index, int) or not 0 <= tree_index < self.spec['tree_count']:
            raise ValueError('Tree index outside forest')
        s = self.spec; tree = s['trees'][tree_index]; records = []
        def visit(node, path, reachable):
            if node < s['internal_count']:
                w = list(tree['split_weights'][node]); b = tree['split_bias'][node]
                constant = not any(v != 0 for v in w)
                records.append(dict(kind='split', index=node, heap=node, path=path, reachable=reachable,
                                    weights=w, bias=b, constant=constant))
                visit(2*node+1, path+'L', reachable and (not constant or b <= 0))
                visit(2*node+2, path+'R', reachable and (not constant or b > 0))
            else:
                leaf = node-s['internal_count']; p = list(tree['leaf_parameters'][leaf])
                records.append(dict(kind='leaf', index=leaf, heap=node, path=path, reachable=reachable,
                    weights=p[:-1] if s['leaf_mode']=='linear' else [0.]*s['feature_count'], bias=p[-1]))
        visit(0, '', True)
        return records

    def format_tree(self, tree_index=0, include_leaves=True):
        s = self.spec
        lines = [f'TREE {tree_index}: depth={s["depth"]}, leaves={s["leaf_mode"]}, input={s["input_space"]}',
                 f'Coefficient display: {self.precision} significant digits; exact zeros omitted.',
                 'At every cut: h > 0 goes RIGHT; h <= 0 goes LEFT (including equality).']
        for r in self.records(tree_index):
            if r['kind']=='leaf' and not include_leaves: continue
            indent = '  '*len(r['path']); label = 'N' if r['kind']=='split' else 'L'
            status = 'reachable' if r['reachable'] else 'UNREACHABLE (constant ancestor)'
            lines.append(f'{indent}{label}{r["index"]} [path={r["path"] or "root"}; heap={r["heap"]}; {status}]')
            function = 'h' if r['kind']=='split' else 'f'
            lines.append(f'{indent}  {function}_{tree_index}_{r["index"]}(x) = '+self.affine(r['weights'], r['bias']))
            if r['kind']=='split':
                def child(j): return 'N'+str(j) if j<s['internal_count'] else 'L'+str(j-s['internal_count'])
                route = f'<= 0 -> {child(2*r["heap"]+1)}; > 0 -> {child(2*r["heap"]+2)}'
                if r['constant']: route += '; CONSTANT: always '+('RIGHT' if r['bias']>0 else 'LEFT')
                lines.append(indent+'  '+route)
        lines.append('Selected leaf f is a class-0 logit contribution; forest signal margin = -sum(f).')
        return '\n'.join(lines)+'\n'

    def print_tree(self, tree_index=0, file=None, include_leaves=True):
        """Write the tree equations to a text stream (stdout by default)."""
        (file if file is not None else sys.stdout).write(self.format_tree(tree_index, include_leaves))

    def format_forest(self):
        return '\n'.join(self.format_tree(i) for i in range(self.spec['tree_count']))
