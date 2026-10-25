"""Exact frozen-support, unpenalized hard-TAO refits.

Masks are selected once from the parent, including biases and every free class.
They are never recomputed from a refitted model. No epsilon pruning is used.
"""
import copy
import hashlib
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from .hard import HardTAOClassifier, cross_entropy_rows, fit_leaf_parameters
from .split_search import best_threshold


def _digest(masks):
    h=hashlib.sha256()
    for group in masks:
        for key in sorted(group):
            a=np.ascontiguousarray(group[key])
            h.update(key.encode());h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


def freeze_support(parent, mode='leaf_only'):
    """Copy a sparse parent without changing its predictions, then remove L1.

    A leaf-only child and a full child can share exactly the same masks. To
    continue from the leaf-only child, use continue_full_refit, not re-freezing.
    """
    if mode not in ('leaf_only','full'):raise ValueError('Unknown support-refit mode')
    if isinstance(parent,FrozenSupportTAO):raise ValueError('Support already frozen; do not select it again')
    model=FrozenSupportTAO.__new__(FrozenSupportTAO)
    model.__dict__.update(copy.deepcopy(parent.__dict__))
    model.support_masks=[];model.support_fixed=[]
    for tree in model.trees:
        for a in (tree.split_w,tree.split_b,tree.leaf_parameters):
            if not np.isfinite(a).all():raise ValueError('Nonfinite parent coefficients')
        model.support_masks.append(dict(split_w=tree.split_w!=0,split_b=tree.split_b!=0,leaf=tree.leaf_parameters!=0))
        model.support_fixed.append(dict(w=tree.split_w.copy(),b=tree.split_b.copy()))
    model.support_sha256=_digest(model.support_masks)
    model.support_fixed_sha256=_digest(model.support_fixed)
    model.support_mode=mode
    model.support_architecture=(model.depth,model.ntrees,model.n_features_in_,model.n_classes_,model.leaf_mode)
    model.parent_penalties=dict(split_alpha=parent.split_alpha,leaf_alpha=parent.leaf_alpha)
    model.split_alpha=model.leaf_alpha=0.
    model.split_search='hard_threshold';model.split_solver='lbfgsb'
    model.assert_support()
    return model


def continue_full_refit(leaf_model):
    """Preserve the original support after leaf fitting, even if a free value became zero."""
    leaf_model.assert_support()
    model=copy.deepcopy(leaf_model);model.support_mode='full';model.assert_support()
    return model


class FrozenSupportTAO(HardTAOClassifier):
    def assert_support(self):
        if self.split_alpha!=0 or self.leaf_alpha!=0:raise ValueError('Support refits must be unpenalized')
        if self.support_mode not in ('leaf_only','full'):raise ValueError('Unknown support mode')
        if (self.depth,self.ntrees,self.n_features_in_,self.n_classes_,self.leaf_mode)!=self.support_architecture:
            raise ValueError('Frozen architecture changed')
        if len(self.trees)!=len(self.support_masks) or len(self.trees)!=len(self.support_fixed):
            raise ValueError('Frozen topology changed')
        if _digest(self.support_masks)!=self.support_sha256 or _digest(self.support_fixed)!=self.support_fixed_sha256:
            raise ValueError('Frozen support contract changed')
        for t,m,f in zip(self.trees,self.support_masks,self.support_fixed):
            if (t.depth,t.features,t.classes,t.leaf_mode,t.internal,t.leaves)!=(
                self.depth,self.n_features_in_,self.n_classes_,self.leaf_mode,2**self.depth-1,2**self.depth):
                raise ValueError('Frozen tree layout changed')
            for a,key in [(t.split_w,'split_w'),(t.split_b,'split_b'),(t.leaf_parameters,'leaf')]:
                if a.shape!=m[key].shape or not np.isfinite(a).all() or np.any(a[~m[key]]!=0):
                    raise ValueError('Nonfinite coefficients, changed shape, or frozen-zero regrowth')
            constant=~m['split_w'].any(axis=1)
            fixed=np.ones(t.internal,dtype=bool) if self.support_mode=='leaf_only' else constant
            if not np.array_equal(t.split_w[fixed],f['w'][fixed]) or not np.array_equal(t.split_b[fixed],f['b'][fixed]):
                raise ValueError('Fixed split or constant routing changed')

    def _inputs(self,*args,**kwargs):
        self.assert_support()
        return super()._inputs(*args,**kwargs)

    def _leaf_trainable_mask(self,tree,leaf=None):
        mask=self.support_masks[self.trees.index(tree)]['leaf']
        return mask if leaf is None else mask[leaf]

    def _global_refit(self,x,y,weights):
        self.assert_support()
        stats=super()._global_refit(x,y,weights)
        self.assert_support()
        return stats

    def support_counts(self):
        self.assert_support()
        return dict(allowed_coefficients=sum(int(v.sum()) for m in self.support_masks for v in m.values()),
            actual_nonzeros=sum(int(np.count_nonzero(a)) for t in self.trees for a in (t.split_w,t.split_b,t.leaf_parameters)),
            mask_sha256=self.support_sha256)

    def _split_update(self,tree,node,x,y,weights,offset,normalizer):
        self.assert_support();mask=self.support_masks[self.trees.index(tree)]
        allowed=np.r_[mask['split_w'][node],mask['split_b'][node]][:,None]
        if self.support_mode=='leaf_only' or not mask['split_w'][node].any():
            return dict(converged=True,accepted=False,fitted=False,iterations=0,solver_seconds=0.,
                        frozen=True,reason='leaf_only' if self.support_mode=='leaf_only' else 'constant_routing')
        losses=[cross_entropy_rows(offset+tree.predict_subtree(x,child,self.indexed),y) for child in (2*node+1,2*node+2)]
        delta=losses[0]-losses[1];active=weights*np.abs(delta)>0
        old_w,old_b=tree.split_w[node].copy(),float(tree.split_b[node])
        def value(w,b):
            if np.any(w[~mask['split_w'][node]]!=0) or (not mask['split_b'][node] and b!=0):
                raise ValueError('Split proposal violates frozen support')
            return float(np.dot(weights,np.where(x@w+b>0,losses[1],losses[0]))/normalizer)
        before=value(old_w,old_b);best=before;best_w,best_b=old_w,old_b
        directions=[('current',old_w)];stats=None;selected='unchanged'
        targets=(delta[active]>0).astype(int)
        if np.array_equal(x[active]@old_w+old_b>0,targets.astype(bool)):
            # The current split already realizes every positive-weight event's
            # lower-loss child. This certifies the reduced HARD objective even
            # if the auxiliary logistic problem has no finite optimum.
            return dict(converged=True,accepted=False,fitted=False,iterations=0,solver_seconds=0.,
                objective_before=before,objective_after=before,split_certificate='pointwise_lower_bound')
        if len(np.unique(targets))==2:
            design=np.column_stack((x[active],np.ones(active.sum())))
            fitted,stats=fit_leaf_parameters(design,targets,None,weights[active]*np.abs(delta[active]),normalizer,
                -np.r_[old_w,old_b][:,None],np.ones(design.shape[1],dtype=bool),0.,self.split_max_iter,
                ftol=getattr(self,'leaf_ftol',1e-12),gtol=getattr(self,'leaf_gtol',1e-8),trainable_mask=allowed)
            w,b=-fitted[:-1,0],float(-fitted[-1,0])
            after=value(w,b)
            if after<best:best,best_w,best_b,selected=after,w,b,'logistic_raw'
            directions.append(('logistic',w))
            directions.extend((f'blend_{f:g}',(1-f)*old_w+f*w) for f in (.25,.5,.75))
        directions.append(('constant',np.zeros_like(old_w)))
        for name,w in directions:
            # A frozen-zero bias cannot be revived by threshold search.
            b=best_threshold(x@w,losses[0],losses[1],weights,normalizer,0.)[0] if mask['split_b'][node] else 0.
            after=value(w,b)
            if after<best:best,best_w,best_b,selected=after,w,b,name
        if best<before:tree.split_w[node],tree.split_b[node]=best_w,best_b
        self.assert_support()
        pointwise=np.array_equal(x[active]@tree.split_w[node]+tree.split_b[node]>0,targets.astype(bool))
        return dict(converged=bool(pointwise or stats is None or stats['converged']),accepted=best<before,fitted=stats is not None,
            iterations=0 if stats is None else stats['iterations'],solver_seconds=0. if stats is None else stats['seconds'],
            objective_before=before,objective_after=best,selected_direction=selected,surrogate=stats,
            split_certificate='pointwise_lower_bound' if pointwise else 'configured_proposals')


def separation_check(model,x,y,sample_weight=None,time_limit=120.):
    """LP check for a logistic recession direction on the current fixed routes.

    Seek d with every true-vs-other class margin >= 0 and at least one > 0.
    Such a direction makes unpenalized CE decrease toward an unattained limit
    (complete OR quasi separation). Box bounds only fix the homogeneous LP's
    scale; they do not constrain fitted coefficients. Null logit gauges give
    zero improvement and do not count as separation. LP failure is inconclusive.
    """
    x,y,weights=model._inputs(x,y,sample_weight)
    positive=weights>0;x,y=x[positive],y[positive]
    design=model._joint_design(x);classes=model.n_classes_;free=classes-1
    mask=np.concatenate([model._leaf_trainable_mask(t).reshape(-1,free) for t in model.trees]).ravel()
    active=np.flatnonzero(mask)
    if not len(active):return dict(status='no_recession_direction',score=0.,constraints=0,variables=0)
    rows=[]
    for other in range(classes):
        indices=np.flatnonzero(y!=other)
        block=design[indices]
        selectors=np.column_stack([(y[indices]==k).astype(float)-(other==k) for k in range(free)])
        # C-order flattening matches theta.ravel() in the leaf solver.
        coo=block.tocoo()
        expanded=sparse.coo_matrix((np.concatenate([coo.data*selectors[coo.row,k] for k in range(free)]),
            (np.tile(coo.row,free),np.concatenate([coo.col*free+k for k in range(free)]))),
            shape=(len(indices),block.shape[1]*free)).tocsr()
        expanded.eliminate_zeros();rows.append(expanded[:,active])
    margins=sparse.vstack(rows,format='csr')
    scales=np.asarray(abs(margins).max(axis=1).toarray()).ravel()
    margins=sparse.diags(1/np.maximum(scales,1.))@margins
    objective=-np.asarray(margins.mean(axis=0)).ravel()
    result=linprog(objective,A_ub=-margins,b_ub=np.zeros(margins.shape[0]),bounds=(-1.,1.),
        method='highs',options=dict(time_limit=float(time_limit),primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    record=dict(constraints=margins.shape[0],variables=len(active),message=result.message)
    if not result.success:return dict(record,status='inconclusive',score=None)
    score=float(-result.fun);minimum=float(np.min(margins@result.x,initial=0.))
    if minimum < -1e-8:return dict(record,status='inconclusive',score=score,min_margin=minimum)
    # Near-zero positive optima are not used to certify finite fitting.
    status='separated' if score>1e-7 else 'inconclusive' if score>1e-10 else 'no_recession_direction'
    return dict(record,status=status,score=score,min_margin=minimum)
