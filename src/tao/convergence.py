"""Numerical stopping checks for hard TAO; no claim of global optimality."""
import numpy as np
from .hard import ce_value_gradient


def leaf_kkt_residual(model, x, y, sample_weight=None):
    """Infinity norm of the normalized CE + L1 leaf stationarity violation."""
    x,y,weights=model._inputs(x,y,sample_weight)
    design=model._joint_design(x)
    theta=np.concatenate([t.leaf_parameters.reshape(-1,model.n_classes_-1) for t in model.trees])
    penalty=np.concatenate([np.tile(t.penalty_rows,t.leaves) for t in model.trees])[:,None]
    _,gradient=ce_value_gradient(theta,design,y,None,weights,weights.sum())
    penalized=np.broadcast_to(penalty,theta.shape)
    nonzero=np.abs(theta)>1e-10
    violation=np.abs(gradient).copy()
    selected=penalized & nonzero
    violation[selected]=np.abs(gradient[selected]+model.leaf_alpha*np.sign(theta[selected]))
    selected=penalized & ~nonzero
    violation[selected]=np.maximum(np.abs(gradient[selected])-model.leaf_alpha,0)
    masks=[model._leaf_trainable_mask(t) for t in model.trees]
    if any(m is not None for m in masks):
        allowed=np.concatenate([np.ones_like(t.leaf_parameters,dtype=bool).reshape(-1,model.n_classes_-1)
            if m is None else m.reshape(-1,model.n_classes_-1) for t,m in zip(model.trees,masks)])
        violation[~allowed]=0  # Frozen coordinates have equality constraints, not zero-gradient conditions.
    return float(violation.max(initial=0))


def convergence_check(profile, kkt, route_change, previous_stable=0,
                      loss_rtol=1e-5, kkt_tol=1e-6, route_tol=1e-4, patience=3):
    before,after=profile['objective_before'],profile['objective_after']
    if not np.isfinite(after) or after>before+1e-8:
        raise ValueError('Training objective increased or is nonfinite')
    relative_gain=max(0,(before-after)/max(abs(before),1e-12))
    inner=all(s['converged'] for s in profile['leaf_fits']+profile['split_fits'])
    inner=inner and profile['global_refit']['converged'] and kkt<=kkt_tol
    stable=bool(inner and relative_gain<=loss_rtol and route_change<=route_tol)
    count=previous_stable+1 if stable else 0
    return dict(relative_objective_gain=relative_gain,leaf_kkt_residual=kkt,
        routing_change_fraction=route_change,inner_fits_converged=bool(inner),
        stable=stable,stable_epochs=count,converged=count>=patience)
