"""Exact-zero and safe structural work counts; no empirical pruning or quantization."""
import numpy as np


def sparsity_profile(model):
    total=nonzero=split_weights=leaf_weights=0
    live_internal=live_leaves=live_nonzero=all_macs=path_macs=path_compares=0
    for tree in model.trees:
        total+=tree.split_w.size+tree.split_b.size+tree.leaf_parameters.size
        nonzero+=int(np.count_nonzero(tree.split_w)+np.count_nonzero(tree.split_b)+np.count_nonzero(tree.leaf_parameters))
        split_weights+=int(np.count_nonzero(tree.split_w));leaf_weights+=int(np.count_nonzero(tree.leaf_parameters[:,:-1]))
        def visit(node):
            nonlocal live_internal,live_leaves,live_nonzero,all_macs
            if node>=tree.internal:
                parameters=tree.leaf_parameters[node-tree.internal]
                macs=int(np.count_nonzero(parameters[:-1]));live_leaves+=1
                live_nonzero+=int(np.count_nonzero(parameters));all_macs+=macs
                return macs,0
            weights=tree.split_w[node];macs=int(np.count_nonzero(weights));live_internal+=1
            live_nonzero+=macs+int(tree.split_b[node]!=0);all_macs+=macs
            if macs==0:
                return visit(2*node+1+int(tree.split_b[node]>0))
            left=visit(2*node+1);right=visit(2*node+2)
            return macs+max(left[0],right[0]),1+max(left[1],right[1])
        macs,compares=visit(0);path_macs+=macs;path_compares+=compares
    return dict(dense_parameter_slots=total,nonzero_coefficients=nonzero,
        nonzero_split_feature_weights=split_weights,nonzero_leaf_feature_weights=leaf_weights,
        zero_coefficients=total-nonzero,structurally_reachable_internal_nodes=live_internal,
        structurally_reachable_leaves=live_leaves,reachable_nonzero_coefficients=live_nonzero,
        reachable_all_node_multiplications=all_macs,selected_path_multiplications_upper_bound=path_macs,
        selected_path_comparisons_upper_bound=path_compares,
        note='Exact zeros only. Unreachable branches removed only for identically constant split decisions. Counts are not FPGA latency/DSP estimates.')
