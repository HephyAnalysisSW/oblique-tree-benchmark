"""Binary floating-point inference interchange for the FPGA implementation handoff."""
import numpy as np


def export_model(model,feature_names,class_names,scaler_mean,scaler_scale,raw=False):
    if model.n_classes_!=2:raise ValueError('FPGA interchange v1 supports binary models only')
    if not (np.all(model.transform.mean==0) and np.all(model.transform.scale==1)):
        raise ValueError('Expected the standardized-input identity adapter')
    mean=np.asarray(scaler_mean,dtype=float);scale=np.asarray(scaler_scale,dtype=float)
    if mean.shape!=(model.n_features_in_,) or scale.shape!=mean.shape or np.any(scale<=0):raise ValueError('Invalid scaler')
    if len(feature_names)!=model.n_features_in_ or len(class_names)!=2:raise ValueError('Invalid feature or class order')
    trees=[]
    for tree in model.trees:
        w=tree.split_w.copy();b=tree.split_b.copy();leaf=tree.leaf_parameters[:,:,0].copy()
        if raw:
            w=w/scale;b=b-w@mean
            if model.leaf_mode=='linear':
                leaf[:,:-1]/=scale;leaf[:,-1]-=leaf[:,:-1]@mean
        arrays=[w,b,leaf]
        if not all(np.isfinite(a).all() for a in arrays):raise ValueError('Nonfinite coefficients')
        trees.append(dict(split_weights=w.tolist(),split_bias=b.tolist(),leaf_parameters=leaf.tolist()))
    return dict(schema='tree-benchmark.binary-hard-oblique.v1',features=list(feature_names),classes=list(class_names),
        input_space='raw' if raw else 'standardized',feature_count=model.n_features_in_,tree_count=model.ntrees,
        depth=model.depth,internal_count=2**model.depth-1,leaf_count=2**model.depth,leaf_mode=model.leaf_mode,
        preprocessing=dict(standardized_equals='(raw - mean) / scale',mean=mean.tolist(),scale=scale.tolist(),folded_into_coefficients=raw),
        routing=dict(right_if='dot(weights,input) + bias > 0',equality='left',left_child='2*node+1',right_child='2*node+2',leaf_index='node-internal_count'),
        score=dict(free_class=0,reference_class=1,forest='sum of selected leaf affine outputs',signal_margin='negative forest output',signal_probability='sigmoid(signal_margin)'),
        quantized=False,coefficient_dtype='IEEE754 binary64',trees=trees)


def predict_margin(spec,inputs):
    """Independent evaluator; returns increasing signal margin, routes, tree outputs."""
    if spec['schema']!='tree-benchmark.binary-hard-oblique.v1':raise ValueError('Unsupported schema')
    x=np.asarray(inputs,dtype=float)
    if x.ndim!=2 or x.shape[1]!=spec['feature_count'] or not np.isfinite(x).all():raise ValueError('Expected finite ordered features')
    outputs=[];assignments=[]
    for tree in spec['trees']:
        w=np.asarray(tree['split_weights'],dtype=float).reshape(spec['internal_count'],spec['feature_count'])
        b=np.asarray(tree['split_bias'],dtype=float);p=np.asarray(tree['leaf_parameters'],dtype=float)
        nodes=np.zeros(len(x),dtype=int)
        for _ in range(spec['depth']):
            # Evaluate one level for the examples currently at each node.
            previous=nodes.copy()
            for node in np.unique(previous):
                rows=np.flatnonzero(previous==node);right=x[rows]@w[node]+b[node]>0
                nodes[rows]=2*node+1+right.astype(int)
        leaf=nodes-spec['internal_count'];value=np.empty(len(x))
        for index in np.unique(leaf):
            rows=np.flatnonzero(leaf==index)
            value[rows]=x[rows]@p[index,:-1]+p[index,-1] if spec['leaf_mode']=='linear' else p[index,0]
        outputs.append(value);assignments.append(leaf)
    return -sum(outputs,np.zeros(len(x))),np.asarray(assignments).T,np.asarray(outputs).T


def deterministic_working_point(margin,y,fpr=.01):
    """Strict margin threshold with at most floor(fpr*Nbackground) false accepts."""
    margin=np.asarray(margin);y=np.asarray(y)
    if margin.ndim!=1 or y.shape!=margin.shape or not np.isfinite(margin).all() or not np.isin(y,[0,1]).all():raise ValueError('Invalid margins or binary labels')
    negative=np.sort(margin[y==0])[::-1]
    if not len(negative) or not np.any(y==1) or not 0<=fpr<1:raise ValueError('Expected both classes and 0<=FPR<1')
    budget=int(np.floor(fpr*len(negative)));threshold=float(negative[budget])
    selected=margin>threshold
    return dict(signal_margin_threshold=threshold,comparison='>',background_count=len(negative),
        maximum_false_positives=budget,false_positives=int(selected[y==0].sum()),
        achieved_background_efficiency=float(selected[y==0].mean()),signal_efficiency=float(selected[y==1].mean()))


def operation_counts(spec):
    n,d,f=spec['tree_count'],spec['depth'],spec['feature_count'];linear=spec['leaf_mode']=='linear'
    width=f+1 if linear else 1
    stored=n*(spec['internal_count']*(f+1)+spec['leaf_count']*width)
    path=n*(d*f+(f if linear else 0))
    return dict(stored_scalar_parameters=stored,selected_path_multiplications=path,
        selected_path_additions=path+max(n-1,0),selected_path_comparisons=n*d,
        all_node_multiplications=n*(spec['internal_count']*f+(spec['leaf_count']*f if linear else 0)),
        arithmetic_note='Dense scalar counts; no zero-weight elimination, DSP mapping, clock or pipeline assumption')


def cpp_header(spec):
    def value(v):
        if isinstance(v,list):return '{'+','.join(value(item) for item in v)+'}'
        return float(v).hex()
    n,f,i,l=spec['tree_count'],spec['feature_count'],spec['internal_count'],spec['leaf_count']
    if not i:raise ValueError('C++ interchange requires depth >= 1')
    width=f+1 if spec['leaf_mode']=='linear' else 1
    text=['#pragma once','// Generated floating-point reference coefficients; not quantized.',
        'namespace oblique_model {',f'constexpr int N_FEATURES={f}, N_TREES={n}, DEPTH={spec["depth"]}, N_INTERNAL={i}, N_LEAVES={l}, LEAF_WIDTH={width};']
    for name,key,dims in [('SPLIT_W','split_weights',f'[{n}][{i}][{f}]'),('SPLIT_B','split_bias',f'[{n}][{i}]'),('LEAF','leaf_parameters',f'[{n}][{l}][{width}]')]:
        text.append(f'constexpr double {name}{dims}='+value([tree[key] for tree in spec['trees']])+';')
    return '\n'.join(text+['}',''])
