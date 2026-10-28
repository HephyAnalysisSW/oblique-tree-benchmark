"""Exact coordinate TAO with axis-aligned cuts and the existing leaf solvers.

A nonconstant cut has one fixed unit coefficient (+1 or -1) and a learned
threshold. Split L1 must be zero: feature selection is discrete, not achieved
by penalizing or rescaling a free hyperplane normal.
"""
import time
import numpy as np
from .hard import HardTAOClassifier, cross_entropy_rows
from .split_search import best_threshold


def assert_axis(model):
    for tree in model.trees:
        counts=np.count_nonzero(tree.split_w,axis=1)
        if np.any(counts>1) or np.any(~np.isin(tree.split_w,[-1.,0.,1.])):
            raise ValueError('Expected axis-aligned unit cuts or input-independent routing')


class AxisTAOClassifier(HardTAOClassifier):
    def initialize(self,features,classes,transform=None):
        if self.split_alpha!=0:
            raise ValueError('Axis TAO requires split_alpha=0; only leaf coefficients use L1')
        super().initialize(features,classes,transform)
        # A valid deterministic axis initialization even before optional CART seeding.
        for tree in self.trees:
            for row in tree.split_w:
                feature=int(np.argmax(np.abs(row)));sign=1. if row[feature]>=0 else -1.
                row[:]=0.;row[feature]=sign
        assert_axis(self)
        return self

    def _split_update(self,tree,node,x,y,weights,offset,normalizer):
        started=time.perf_counter()
        if self.split_alpha!=0:raise ValueError('Axis split penalty must remain zero')
        old=tree.split_w[node]
        if np.count_nonzero(old)>1 or np.any(~np.isin(old,[-1.,0.,1.])):
            raise ValueError('Cannot resume an oblique node as an axis cut')
        losses=[cross_entropy_rows(offset+tree.predict_subtree(x,child,self.indexed),y)
                for child in (2*node+1,2*node+2)]
        delta=losses[0]-losses[1];active=weights*np.abs(delta)>0
        def value(w,b):
            return float(np.dot(weights,np.where(x@w+b>0,losses[1],losses[0]))/normalizer)
        before=value(old,tree.split_b[node]);best=before
        best_w,best_b=old.copy(),float(tree.split_b[node]);chosen='unchanged';candidates=0
        # No active event can benefit from rerouting. Keep the exact current cut.
        if active.any():
            for feature in range(x.shape[1]):
                for sign in (1.,-1.):
                    bias,info=best_threshold(sign*x[:,feature],losses[0],losses[1],weights,normalizer,0.)
                    w=np.zeros(x.shape[1]);w[feature]=sign
                    after=value(w,bias)  # Verify the actual representable hard routing.
                    candidates+=info['threshold_candidates']
                    if after<best:
                        best,best_w,best_b=after,w,bias;chosen=f'feature_{feature}_sign_{sign:g}'
            # Include input-independent decisions, including unequal child losses.
            w=np.zeros(x.shape[1])
            bias,info=best_threshold(np.zeros(len(x)),losses[0],losses[1],weights,normalizer,0.)
            after=value(w,bias);candidates+=info['threshold_candidates']
            if after<best:best,best_w,best_b,chosen=after,w,bias,'constant'
        if best<before:tree.split_w[node],tree.split_b[node]=best_w,best_b
        return dict(events=len(y),active_events=int(active.sum()),tied_events=int((delta==0).sum()),
            fitted=bool(active.any()),accepted=best<before,converged=True,iterations=0,solver_seconds=0.,
            objective_before=before,objective_after=best,search='axis_exact_threshold',
            selected_direction=chosen,threshold_candidates=candidates,search_seconds=time.perf_counter()-started)


def axis_parameter_counts(model):
    """Separate learned scalar values from discrete cut choices and stored units."""
    assert_axis(model)
    internal=sum(t.internal for t in model.trees)
    nonconstant=sum(int(np.count_nonzero(np.count_nonzero(t.split_w,axis=1))) for t in model.trees)
    leaf_slots=sum(t.leaf_parameters.size for t in model.trees)
    leaf_nonzero=sum(int(np.count_nonzero(t.leaf_parameters)) for t in model.trees)
    threshold_nonzero=sum(int(np.count_nonzero(t.split_b[np.any(t.split_w!=0,axis=1)])) for t in model.trees)
    all_leaf_products=path_leaf_products=path_comparisons=0
    for tree in model.trees:
        def visit(node):
            nonlocal all_leaf_products
            if node>=tree.internal:
                products=int(np.count_nonzero(tree.leaf_parameters[node-tree.internal,:-1]))
                all_leaf_products+=products
                return products,0
            if not np.any(tree.split_w[node]):
                return visit(2*node+1+int(tree.split_b[node]>0))
            left,right=visit(2*node+1),visit(2*node+2)
            return max(left[0],right[0]),1+max(left[1],right[1])
        products,comparisons=visit(0)
        path_leaf_products+=products;path_comparisons+=comparisons
    return dict(continuous_slots=internal+leaf_slots,leaf_coefficient_slots=leaf_slots,
        leaf_nonzero_coefficients=leaf_nonzero,nonzero_active_thresholds=threshold_nonzero,
        learned_nonzero_scalars=leaf_nonzero+threshold_nonzero,feature_index_slots=internal,
        direction_bit_slots=internal,nonconstant_cuts=nonconstant,routing_multiplications=0,
        reachable_all_leaf_multiplications=all_leaf_products,selected_path_leaf_multiplications_bound=path_leaf_products,
        selected_path_comparisons_bound=path_comparisons,
        note='Fixed +/-1 routing coefficients are not learned scalars. Feature indices/direction bits and constant-routing flags need separate storage; no FPGA resource estimate.')
