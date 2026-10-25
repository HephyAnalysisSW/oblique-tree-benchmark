"""Paired, class-stratified validation bootstrap for fixed-score ROC working points.

Intervals condition on the fitted/selected models. They do not account for
training seeds or hyperparameter/checkpoint selection on the validation set.
"""
import numpy as np


class WeightedWorkingPoint:
    def __init__(self,y,scores,fpr=.01):
        y=np.asarray(y);scores=np.asarray(scores,dtype=float)
        if y.ndim!=1 or scores.shape!=y.shape or not np.isfinite(scores).all():raise ValueError('Finite aligned binary scores required')
        if set(np.unique(y))!={0,1} or not 0<fpr<1:raise ValueError('Require both binary classes and interior working point')
        self.y=y;self.fpr=fpr;self.order=np.argsort(-scores,kind='stable')
        ordered=scores[self.order];self.groups=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
        self.signal=y[self.order]==1

    def evaluate(self,counts):
        counts=np.asarray(counts)
        if counts.shape!=self.y.shape or not np.isfinite(counts).all() or np.any(counts<0):raise ValueError('Finite nonnegative aligned resample counts required')
        ordered=counts[self.order]
        positive=np.cumsum(np.add.reduceat(ordered*self.signal,self.groups))
        negative=np.cumsum(np.add.reduceat(ordered*~self.signal,self.groups))
        if positive[-1]<=0 or negative[-1]<=0:raise ValueError('Each class needs positive resample mass')
        return float(np.interp(self.fpr,np.r_[0.,negative/negative[-1]],np.r_[0.,positive/positive[-1]]))


def paired_intervals(y,scores,reference='XGBoost',repeats=1000,seed=20261025,fpr=.01,comparisons=None):
    """Reuse each stratified bootstrap sample across all models, including tied scores."""
    if reference not in scores or repeats<20:raise ValueError('Reference scores and at least 20 replicates required')
    y=np.asarray(y);calculators={name:WeightedWorkingPoint(y,value,fpr) for name,value in scores.items()}
    indices=[np.flatnonzero(y==k) for k in [0,1]];rng=np.random.default_rng(seed)
    points={name:calc.evaluate(np.ones(len(y))) for name,calc in calculators.items()}
    estimates={name:np.empty(repeats) for name in scores};counts=np.empty(len(y),dtype=np.int64)
    probabilities=[np.full(len(i),1/len(i)) for i in indices]
    for repeat in range(repeats):
        for group,probability in zip(indices,probabilities):counts[group]=rng.multinomial(len(group),probability)
        for name,calc in calculators.items():estimates[name][repeat]=calc.evaluate(counts)
    models={}
    for name,values in estimates.items():
        delta=values-estimates[reference]
        models[name]=dict(efficiency=points[name],interval95=np.quantile(values,[.025,.975]).tolist(),
            difference_vs_reference=points[name]-points[reference],difference_interval95=np.quantile(delta,[.025,.975]).tolist())
    contrasts={}
    for name,(target,baseline) in (comparisons or {}).items():
        if target not in estimates or baseline not in estimates:raise ValueError('Unknown comparison model')
        contrasts[name]=dict(target=target,baseline=baseline,difference=points[target]-points[baseline],
            interval95=np.quantile(estimates[target]-estimates[baseline],[.025,.975]).tolist())
    return dict(method='paired class-stratified percentile bootstrap of interpolated ROC efficiency',
        reference=reference,repeats=repeats,seed=seed,fixed_fpr=fpr,models=models,comparisons=contrasts,numpy_version=np.__version__,
        note='Conditional validation sampling uncertainty only; no correction for checkpoint/hyperparameter selection or training-seed variation. Test not evaluated.')
