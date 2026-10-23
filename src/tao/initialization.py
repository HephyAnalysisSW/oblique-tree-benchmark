"""Training-only initial partitions and function-preserving forest growth."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .hard import HardTree


def initialize_cart(model,x,y):
    """Seed oblique trees from CART; bootstrap within training for forests.

    Stored class logits use the final class as reference, as in HardTAO.
    Unused branches of an early CART leaf remain inactive and retain zero logits.
    No validation examples or new train/validation split enter initialization.
    """
    rng=np.random.default_rng(model.seed)
    records=[]
    for index,tree in enumerate(model.trees):
        sample=np.arange(len(y)) if model.ntrees==1 else rng.integers(len(y),size=len(y))
        cart=DecisionTreeClassifier(max_depth=max(1,model.depth),random_state=model.seed+index)
        cart.fit(x[sample],y[sample])
        tree.split_w[:]=0;tree.split_b[:]=0;tree.leaf_parameters[:]=0
        def visit(source,target,level):
            if level==model.depth or cart.tree_.children_left[source]<0:
                # Pad early terminal branches by routing left through zero tests.
                terminal=target
                for _ in range(model.depth-level):terminal=2*terminal+1
                counts=np.zeros(model.n_classes_)
                counts[cart.classes_.astype(int)]=cart.tree_.value[source,0]
                # sklearn versions may store fractions; recover counts first.
                counts=counts/counts.sum()*cart.tree_.weighted_n_node_samples[source]
                probabilities=(counts+.5)/(counts.sum()+.5*model.n_classes_)
                tree.leaf_parameters[terminal-tree.internal,-1]=np.log(probabilities[:-1]/probabilities[-1])/model.ntrees
                return
            tree.split_w[target,cart.tree_.feature[source]]=1.
            tree.split_b[target]=-cart.tree_.threshold[source]
            visit(cart.tree_.children_left[source],2*target+1,level+1)
            visit(cart.tree_.children_right[source],2*target+2,level+1)
        visit(0,0,0)
        records.append(dict(tree=index,bootstrap=model.ntrees>1,cart_nodes=int(cart.tree_.node_count)))
    return records


def append_zero_tree(model,target_trees,seed):
    """Append exactly one zero-logit tree, preserving all current predictions."""
    if target_trees!=model.ntrees+1:raise ValueError('Sequential growth must append exactly one tree')
    rng=np.random.default_rng(seed+1009*target_trees)
    model.trees.append(HardTree(model.depth,model.n_features_in_,model.n_classes_,model.leaf_mode,rng))
    model.ntrees=target_trees
