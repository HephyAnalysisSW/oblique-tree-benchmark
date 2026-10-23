"""Train hard oblique forests with linear multiclass leaves on fixed partitions."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import time
import numpy as np
from data_utils import load_config, load_metadata, resolve, write_json
from evaluate import compute_metrics
from tao.hard import HardTAOClassifier
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['qg_vs_wzt', 'qg_vs_top', 'multiclass'], default='qg_vs_wzt')
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--trees', type=int, required=True)
    parser.add_argument('--leaf-mode', choices=['linear', 'constant'], default='linear')
    parser.add_argument('--split-alpha', type=float, default=1e-5)
    parser.add_argument('--leaf-alpha', type=float, default=1e-5)
    parser.add_argument('--leaf-max-iter', type=int, default=100)
    parser.add_argument('--split-max-iter', type=int, default=100)
    parser.add_argument('--split-solver', choices=['liblinear', 'lbfgsb'], default='liblinear')
    parser.add_argument('--routing', choices=['dense', 'indexed', 'batched'], default='dense')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--plot-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.plot_every, args.leaf_max_iter, args.split_max_iter) < 1:
        parser.error('Iteration counts and plot interval must be positive')
    if args.suffix and not all(c.isalnum() or c in '_-' for c in args.suffix):
        parser.error('Suffix must contain only letters, numbers, hyphens, or underscores')
    run = (f'{args.task}_hardtao_{args.leaf_mode}_d{args.depth}_n{args.trees}'
           f'_sa{args.split_alpha:g}_la{args.leaf_alpha:g}_seed{args.seed}')
    if args.split_solver != 'liblinear':
        run += '_' + args.split_solver
    if args.suffix:
        run += '_' + args.suffix
    model_path = resolve(f'models/{run}.pkl')
    if model_path.exists():
        raise FileExistsError(f'Refusing to overwrite {model_path}')
    reference = load_config(f'configs/bdt_{args.task}_seed42.yaml')
    metadata = load_metadata(reference['data']['metadata_path'])
    with np.load(resolve(reference['data']['dataset_path'])) as arrays:
        x, y = arrays['X_train'].astype(np.float64), arrays['y_train']
        xv, yv = arrays['X_val'].astype(np.float64), arrays['y_val']
    # Do not load test arrays during training or model selection.
    with open(resolve(reference['model']['output_model_path']), 'rb') as handle:
        bdt = pickle.load(handle)['model']
    reference_metrics, _ = compute_metrics(yv, bdt.predict_proba(xv), metadata['class_names'])
    model = HardTAOClassifier(args.depth, args.trees, args.leaf_mode, args.seed,
                             args.split_alpha, args.leaf_alpha,
                             args.leaf_max_iter, args.split_max_iter,
                             indexed=args.routing, split_solver=args.split_solver).initialize(x.shape[1], len(metadata['class_names']))
    config = dict(run_name=run, seed=args.seed, data=reference['data'],
                  implementation_sha256=hashlib.sha256(resolve('src/tao/hard.py').read_bytes()).hexdigest(),
                  model=dict(type='hard_tao_cross_entropy', output_model_path=f'models/{run}.pkl',
                             tao=vars(args), objective='mean_cross_entropy_plus_l1',
                             split_solver=args.split_solver + '_l1', leaf_solver='lbfgsb_exact_l1_parameterization',
                             preprocessing='identity_on_fixed_standardized_arrays',
                             logits='K-1 with last class reference', global_refit='all_training_events'),
                  evaluation=dict(output_prefix=f'results/{run}', fixed_fpr=0.01))
    write_json(f'configs/{run}.yaml', config)
    print(json.dumps(dict(run=run, training_events=len(y), validation_events=len(yv),
                          classes=model.n_classes_, reference_validation=reference_metrics)), flush=True)
    history, best_auc, best_epoch = [], -np.inf, -1
    started = time.perf_counter()

    def plot_history():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r['validation']['macro_auc'] for r in history], label='Hard TAO')
        axes[0].axhline(reference_metrics['macro_auc'], ls='--', color='tab:orange', label='BDT')
        axes[0].set_ylabel('Validation macro AUC')
        axes[1].plot(epochs, [r['profile']['objective_after'] for r in history], label='Training objective')
        axes[1].set_ylabel('Mean cross-entropy + L1 penalties')
        for ax in axes:
            ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend()
        fig.suptitle(f'{args.task}: hard {args.leaf_mode} leaves, depth {args.depth}, {args.trees} trees')
        fig.tight_layout()
        fig.savefig(resolve(f'results/{run}_learning.png'), dpi=160)
        plt.close(fig)

    for epoch in range(args.epochs):
        print(f'Starting epoch {epoch + 1}', flush=True)
        profile = model.fit_epoch(x, y)
        if profile['objective_after'] > profile['objective_before'] + 1e-9:
            raise RuntimeError('Forest objective increased; investigate before continuing')
        tick = time.perf_counter()
        validation, _ = compute_metrics(yv, model.predict_proba(xv), metadata['class_names'])
        row = dict(epoch=epoch + 1, profile=profile, validation=validation,
                   validation_seconds=time.perf_counter() - tick)
        history.append(row)
        print(json.dumps(dict(epoch=epoch + 1, seconds=profile['seconds'],
                              objective_before=profile['objective_before'], objective_after=profile['objective_after'],
                              validation_auc=validation['macro_auc'], validation_accuracy=validation['accuracy'],
                              validation_efficiency=validation['signal_efficiency_at_fixed_fpr'],
                              leaf_fits_not_converged=sum(not s['converged'] for s in profile['leaf_fits']),
                              split_fits_not_converged=sum(not s['converged'] for s in profile['split_fits']),
                              global_refit_converged=profile['global_refit']['converged'])), flush=True)
        if validation['macro_auc'] > best_auc + 1e-6:
            best_auc, best_epoch = validation['macro_auc'], epoch
            temporary = model_path.with_suffix('.pkl.tmp')
            with temporary.open('wb') as handle:
                pickle.dump(dict(model=model, config=config, metadata=metadata, history=history), handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(model_path)
            write_json(f'results/{run}_validation.json', dict(run_name=run, best_epoch=epoch + 1,
                validation=validation, reference_validation=reference_metrics,
                complexity=model.complexity(), test_evaluated=False))
        write_json(f'results/{run}_history.json', history)
        if (epoch + 1) % args.plot_every == 0:
            plot_history()
        if epoch - best_epoch >= args.patience:
            break
    plot_history()
    print(json.dumps(dict(run_name=run, best_epoch=best_epoch + 1, epochs_run=len(history),
                          total_seconds=time.perf_counter() - started, test_evaluated=False)), flush=True)


if __name__ == '__main__':
    main()
