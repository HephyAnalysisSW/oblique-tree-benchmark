"""Compare binary runs on the same fixed test partition using shared metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_config, load_metadata, resolve, write_json
from evaluate import compute_metrics, normalize_probabilities, safe_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--reference-label", default="Supplied BDT")
    parser.add_argument("--candidate-label", default="Retrained BDT")
    args = parser.parse_args()
    reference = load_config(args.reference_config)
    candidate = load_config(args.candidate_config)
    for key in ("dataset_path", "metadata_path"):
        if resolve(reference["data"][key]) != resolve(candidate["data"][key]):
            raise ValueError(f"Runs must use the same {key}")
    fixed_fpr = float(reference["evaluation"].get("fixed_fpr", 0.01))
    if fixed_fpr != float(candidate["evaluation"].get("fixed_fpr", 0.01)):
        raise ValueError("Runs must use the same fixed FPR")
    metadata = load_metadata(reference["data"]["metadata_path"])
    classes = metadata["class_names"]
    if len(classes) != 2:
        raise ValueError("This comparison plots binary signal ROC curves")
    with np.load(resolve(reference["data"]["dataset_path"])) as arrays:
        y = arrays["y_test"]
    metrics, curves, probabilities = [], [], []
    for config in (reference, candidate):
        prefix = config["evaluation"]["output_prefix"]
        p = normalize_probabilities(np.load(resolve(prefix + "_probabilities.npy")))
        if p.shape != (len(y), 2) or not np.isfinite(p).all():
            raise ValueError("Expected finite two-class probabilities for every test event")
        m, roc = compute_metrics(y, p, classes, fixed_fpr=fixed_fpr)
        probabilities.append(p)
        metrics.append(m)
        curves.append(roc)
    keys = ("accuracy", "macro_auc", "signal_auc", "cross_entropy",
            "expected_calibration_error", "signal_efficiency_at_fixed_fpr")
    summary = {
        "reference_run": reference["run_name"],
        "candidate_run": candidate["run_name"],
        "dataset_path": reference["data"]["dataset_path"],
        "test_events": int(len(y)),
        "fixed_fpr": fixed_fpr,
        "reference_metrics": metrics[0],
        "candidate_metrics": metrics[1],
        "candidate_minus_reference": {key: metrics[1][key] - metrics[0][key] for key in keys},
        "max_absolute_probability_difference": float(np.max(np.abs(probabilities[1] - probabilities[0]))),
        "mean_absolute_probability_difference": float(np.mean(np.abs(probabilities[1] - probabilities[0]))),
        "changed_predicted_labels": int(np.sum(probabilities[1].argmax(axis=1) != probabilities[0].argmax(axis=1))),
    }
    output = resolve(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(output) + "_comparison.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    signal = safe_label(classes[1])
    for label, color, style, width, m, roc in zip(
        (args.reference_label, args.candidate_label), ("#2366aa", "#df761b"),
        ("-", "--"), (2.5, 1.7), metrics, curves
    ):
        fpr, tpr = roc[signal + "_fpr"], roc[signal + "_tpr"]
        for ax in axes:
            ax.plot(fpr, tpr, color=color, linestyle=style, linewidth=width,
                    label=f"{label} (AUC {m['signal_auc']:.6f})")
            ax.plot(fixed_fpr, m["signal_efficiency_at_fixed_fpr"],
                    marker="o", markersize=5, color=color)
    for ax in axes:
        random_fpr = np.geomspace(1e-5, 1, 500)
        ax.plot(random_fpr, random_fpr, color="0.65", linewidth=1, linestyle=":", label="Random")
        ax.axvline(fixed_fpr, color="0.6", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Background acceptance (false positive rate)")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylabel("Signal efficiency (true positive rate)")
    axes[0].set_title("Full ROC curve")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_xlim(1e-4, 1)
    axes[1].set_title("Low background acceptance")
    efficiencies = [m["signal_efficiency_at_fixed_fpr"] for m in metrics]
    axes[1].text(0.04, 0.96,
                 f"At {fixed_fpr:.0%} background acceptance:\n"
                 f"{args.reference_label}: {efficiencies[0]:.4%}\n"
                 f"{args.candidate_label}: {efficiencies[1]:.4%}",
                 transform=axes[1].transAxes, va="top", fontsize=9,
                 bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.95})
    fig.suptitle(f"{classes[0]} background vs {classes[1]} signal | Classifier comparison", fontsize=11)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(str(output) + "_roc." + extension, dpi=180)
    plt.close(fig)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("reference_metrics", "candidate_metrics")}, indent=2))


if __name__ == "__main__":
    main()
