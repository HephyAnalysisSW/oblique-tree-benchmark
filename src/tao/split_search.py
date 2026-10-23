"""Exact hard-loss threshold search along a fixed oblique direction."""
import numpy as np


def best_threshold(scores, loss_left, loss_right, weights, normalizer, alpha):
    """Minimize routing loss + alpha*abs(bias) with right iff score+bias>0.

    Equal scores move together. In each threshold interval choose the bias
    closest to zero, including constant-routing endpoints. Direction weights
    are fixed and their L1 penalty can be added by the caller.
    """
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected nonempty finite one-dimensional scores')
    order=np.argsort(scores,kind='stable');sorted_scores=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(sorted_scores)>0),len(scores)-1]
    unique=sorted_scores[ends]
    # Threshold t in [unique[i], unique[i+1]) makes the first i+1 groups left.
    thresholds=np.empty(len(unique)+1)
    thresholds[0]=0. if unique[0]>0 else np.nextafter(unique[0],-np.inf)
    thresholds[-1]=0. if unique[-1]<=0 else unique[-1]
    if len(unique)>1:
        lo,hi=unique[:-1],unique[1:]
        thresholds[1:-1]=np.where(lo>0,lo,np.where(hi>0,0.,np.nextafter(hi,-np.inf)))
    delta=np.asarray(weights)*(np.asarray(loss_left)-np.asarray(loss_right))
    losses=np.dot(weights,loss_right)+np.r_[0.,np.cumsum(delta[order])[ends]]
    values=losses/normalizer+alpha*np.abs(thresholds)
    finite=np.isfinite(thresholds)&np.isfinite(values)
    index=int(np.argmin(np.where(finite,values,np.inf)))
    if not finite[index]:raise ValueError('No representable threshold')
    return -float(thresholds[index]),dict(threshold_candidates=int(finite.sum()),
        routing_objective=float(values[index]))
