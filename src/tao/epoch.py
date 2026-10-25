"""Serializable node-by-node execution of the existing hard TAO epoch."""
import time
import numpy as np
from .hard import cross_entropy_rows,fit_leaf_parameters


def begin_epoch(model,x,y,sample_weight=None):
    x,y,weights=model._inputs(x,y,sample_weight)
    started=time.perf_counter()
    cached=[tree.predict_subtree(x,indexed=model.indexed) for tree in model.trees]
    total=sum(cached,np.zeros((len(y),model.n_classes_)))
    profile=dict(leaf_seconds=0.,routing_seconds=0.,split_seconds=0.,prediction_seconds=time.perf_counter()-started,
        leaf_fits=[],split_fits=[],objective_before=float(np.dot(weights,cross_entropy_rows(total,y))/weights.sum()+model._penalty()),seconds=0.)
    return dict(tree_index=0,stage='tree_start',cached=cached,total=total,profile=profile)


def step_epoch(model,work,x,y,sample_weight=None):
    """Perform one node update or bookkeeping step; return True when finished.

    The cache, event order, bottom-up node order and numerical solvers match
    HardTAOClassifier.fit_epoch(global_refit=False). Pickle model and work
    together between calls to resume without repeating a partial epoch.
    """
    x,y,weights=model._inputs(x,y,sample_weight);normalizer=weights.sum()
    started=time.perf_counter();profile=work['profile'];stage=work['stage']
    tree=model.trees[work['tree_index']] if stage!='finish' else None
    if stage=='tree_start':
        work['offset']=work['total']-work['cached'][work['tree_index']]
        work['routes']=tree.routes(x,model.indexed)
        work['leaf_index']=0;work['stage']='leaf'
        profile['routing_seconds']+=time.perf_counter()-started
    elif stage=='leaf':
        leaf=work['leaf_index'];rows=work['routes'][tree.internal+leaf]
        if len(rows)==0 or weights[rows].sum()==0:tree.leaf_parameters[leaf]=0
        else:
            fitted,stats=fit_leaf_parameters(tree.design(x[rows]),y[rows],work['offset'][rows],weights[rows],normalizer,
                tree.leaf_parameters[leaf],tree.penalty_rows,model.leaf_alpha,model.leaf_max_iter,
                ftol=getattr(model,'leaf_ftol',1e-10),gtol=getattr(model,'leaf_gtol',1e-7),
                trainable_mask=model._leaf_trainable_mask(tree,leaf))
            tree.leaf_parameters[leaf]=fitted;profile['leaf_fits'].append(stats)
        work['leaf_index']+=1
        if work['leaf_index']==tree.leaves:
            work['node_index']=tree.internal-1;work['stage']='split' if tree.internal else 'tree_end'
        profile['leaf_seconds']+=time.perf_counter()-started
    elif stage=='split':
        node=work['node_index'];rows=work['routes'][node]
        if len(rows):profile['split_fits'].append(model._split_update(tree,node,x[rows],y[rows],weights[rows],work['offset'][rows],normalizer))
        work['node_index']-=1
        if work['node_index']<0:work['stage']='tree_end'
        profile['split_seconds']+=time.perf_counter()-started
    elif stage=='tree_end':
        index=work['tree_index'];work['cached'][index]=tree.predict_subtree(x,indexed=model.indexed)
        work['total']=work.pop('offset')+work['cached'][index];work.pop('routes')
        profile['prediction_seconds']+=time.perf_counter()-started
        work['tree_index']+=1;work['stage']='finish' if work['tree_index']==model.ntrees else 'tree_start'
    elif stage=='finish':
        profile['objective_after']=model._objective_prepared(x,y,weights)
        profile['seconds']+=time.perf_counter()-started
        return True
    else:raise ValueError('Unknown epoch stage: '+stage)
    profile['seconds']+=time.perf_counter()-started
    return False
