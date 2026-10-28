"""Read-only transforms and arithmetic accounting for the two leaf candidates."""
import copy
import math
import numpy as np


def normalized(spec):
    """Positive power-of-two cut scaling; leaves and canonical inputs unchanged.

    Scale each affine cut by the same power of two, including its bias.
    ldexp avoids constructing an overflowing scale for subnormal coefficients.
    This is an export transform, not retraining or pruning.
    """
    out = copy.deepcopy(spec)
    for tree in out['trees']:
        for i, (weights, bias) in enumerate(zip(tree['split_weights'], tree['split_bias'])):
            bound = max(map(abs, [*weights, bias]))
            exponent = math.frexp(bound)[1] if bound else 0
            nonzero = [abs(v) for v in [*weights, bias] if v]
            # Do not flush a nonzero binary64 coefficient when scaling down.
            if nonzero:
                exponent = min(exponent, min(math.frexp(v)[1] for v in nonzero)+1073)
            tree['split_weights'][i] = [math.ldexp(v, -exponent) for v in weights]
            tree['split_bias'][i] = math.ldexp(bias, -exponent)
    return out


def conifer_constant(spec):
    if spec['leaf_mode'] != 'constant':
        raise ValueError('Stock conifer has scalar leaves; linear leaves require an extension')
    spec = normalized(spec)
    internal, leaves, features = spec['internal_count'], spec['leaf_count'], spec['feature_count']
    trees = []
    for tree in spec['trees']:
        # TAO goes left for w.x+b <= 0; conifer uses w.x <= threshold.
        trees.append([dict(feature=[0]*internal+[-2]*leaves,
            weight=tree['split_weights']+[[0.]*features for _ in range(leaves)],
            threshold=[-v for v in tree['split_bias']]+[0.]*leaves,
            value=[0.]*internal+[-p[0] for p in tree['leaf_parameters']],
            children_left=[2*i+1 for i in range(internal)]+[-1]*leaves,
            children_right=[2*i+2 for i in range(internal)]+[-1]*leaves)])
    return dict(n_classes=2, n_features=features, n_trees=spec['tree_count'],
        max_depth=spec['depth'], init_predict=[0.], norm=1., library='tree_benchmark',
        splitting_convention='<=', feature_map={f:i for i,f in enumerate(spec['features'])}, trees=trees)


def arithmetic(spec):
    """Exact-zero sparse counts; selected path and all-reachable-node bounds."""
    feature_weights = leaf_weights = biases = 0
    reachable_nonzero = all_multiplications = path_multiplications = comparisons = path_comparisons = 0
    dense = 0
    for tree in spec['trees']:
        w = np.asarray(tree['split_weights']); b = np.asarray(tree['split_bias']); p = np.asarray(tree['leaf_parameters'])
        linear = spec['leaf_mode'] == 'linear'
        feature_weights += int(np.count_nonzero(w))
        leaf_weights += int(np.count_nonzero(p[:,:-1])) if linear else 0
        biases += int(np.count_nonzero(b)) + int(np.count_nonzero(p[:,-1]))
        dense += w.size+b.size+p.size
        def visit(node):
            nonlocal reachable_nonzero, all_multiplications, comparisons
            if node >= spec['internal_count']:
                row = p[node-spec['internal_count']]
                products = int(np.count_nonzero(row[:-1])) if linear else 0
                reachable_nonzero += int(np.count_nonzero(row)); all_multiplications += products
                return products, 0
            nonzeros = int(np.count_nonzero(w[node]))
            reachable_nonzero += nonzeros + int(b[node] != 0)
            all_multiplications += nonzeros
            if nonzeros == 0:
                return visit(2*node+1+int(b[node]>0))
            comparisons += 1
            children = [visit(2*node+1), visit(2*node+2)]
            return nonzeros+max(v[0] for v in children), 1+max(v[1] for v in children)
        products, path_comp = visit(0)
        path_multiplications += products
        path_comparisons += path_comp
    nonzero = feature_weights+leaf_weights+biases
    return dict(dense_slots=dense, nonzero_coefficients=nonzero, reachable_nonzero_coefficients=reachable_nonzero,
        split_feature_weights=feature_weights, leaf_feature_weights=leaf_weights, biases_or_constants=biases,
        reachable_all_node_multiplications=all_multiplications, reachable_all_node_comparisons=comparisons,
        sparse_path_multiplications_bound=path_multiplications, sparse_path_comparisons_bound=path_comparisons,
        coefficient_value_bits_binary64=64*nonzero,
        note='Coefficient payload excludes topology/indices/padding. Arithmetic counts are not DSP/LUT counts or latency.')
