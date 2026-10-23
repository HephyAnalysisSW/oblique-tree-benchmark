from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from data_utils import load_config, load_dataset, resolve, write_json
import common.syncer  # register plotting outputs for certificate-authenticated EOS upload


DEFAULT_FIXED_FPRS = (0.001, 0.01, 0.1)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(error)


def confidence_coverage(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, dict[str, float | None]]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    result = {}
    for threshold in (0.5, 0.7, 0.8, 0.9, 0.95):
        selected = confidence >= threshold
        result[str(threshold)] = {
            "coverage": float(selected.mean()),
            "accuracy": float((predictions[selected] == y_true[selected]).mean()) if selected.any() else None,
        }
    return result


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        probabilities = np.stack([1.0 - probabilities, probabilities], axis=1)
    if probabilities.shape[1] == 1:
        signal = probabilities[:, 0]
        probabilities = np.stack([1.0 - signal, signal], axis=1)
    return probabilities.astype(np.float32, copy=False)


def safe_label(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def predict_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return normalize_probabilities(model.predict_proba(x))
    if hasattr(model, "predict"):
        return normalize_probabilities(model.predict(x))
    raise TypeError("Model must provide predict_proba(X) or predict(X).")


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    fixed_fpr: float = 0.01,
    fixed_fprs: tuple[float, ...] = DEFAULT_FIXED_FPRS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    probabilities = normalize_probabilities(probabilities)
    class_count = probabilities.shape[1]
    predictions = probabilities.argmax(axis=1)
    binary_targets = np.eye(class_count, dtype=np.int8)[y_true]
    per_class_auc = {
        name: float(roc_auc_score(binary_targets[:, index], probabilities[:, index]))
        for index, name in enumerate(class_names)
    }
    per_class_average_precision = {
        name: float(average_precision_score(binary_targets[:, index], probabilities[:, index]))
        for index, name in enumerate(class_names)
    }
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_auc": float(roc_auc_score(binary_targets, probabilities, average="macro")),
        "per_class_auc": per_class_auc,
        "macro_average_precision": float(average_precision_score(binary_targets, probabilities, average="macro")),
        "per_class_average_precision": per_class_average_precision,
        "cross_entropy": float(log_loss(y_true, probabilities, labels=np.arange(class_count))),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - binary_targets) ** 2, axis=1))),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "confidence_coverage": confidence_coverage(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=np.arange(class_count)).tolist(),
        "fixed_fpr": float(fixed_fpr),
        "fixed_fpr_points": [float(value) for value in fixed_fprs],
        "signal_efficiency_at_fixed_fpr": None,
        "per_class_efficiency_at_fixed_fpr": {},
        "per_class_efficiency_points": {},
    }
    roc = {}
    for index, name in enumerate(class_names):
        fpr, tpr, thresholds = roc_curve(binary_targets[:, index], probabilities[:, index])
        safe_name = safe_label(name)
        roc[f"{safe_name}_fpr"] = fpr
        roc[f"{safe_name}_tpr"] = tpr
        roc[f"{safe_name}_thresholds"] = thresholds
        metrics["per_class_efficiency_at_fixed_fpr"][name] = float(np.interp(fixed_fpr, fpr, tpr))
        metrics["per_class_efficiency_points"][name] = {
            str(value): float(np.interp(value, fpr, tpr)) for value in fixed_fprs
        }
    if class_count == 2:
        signal_fpr = roc[f"{safe_label(class_names[1])}_fpr"]
        signal_tpr = roc[f"{safe_label(class_names[1])}_tpr"]
        metrics["signal_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        metrics["signal_efficiency_at_fixed_fpr"] = float(np.interp(fixed_fpr, signal_fpr, signal_tpr))
    else:
        metrics["signal_auc"] = None
    return metrics, roc


def plot_roc_curves(
    roc: dict[str, np.ndarray],
    per_class_auc: dict[str, float],
    output_path: str | Path,
) -> None:
    output_path = resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    class_items = list(per_class_auc.items())
    if len(class_items) == 2:
        # Binary benchmark convention: plot only the positive/signal class.
        class_items = [class_items[1]]
    for class_name, auc in class_items:
        safe_name = safe_label(class_name)
        ax.plot(
            roc[f"{safe_name}_fpr"],
            roc[f"{safe_name}_tpr"],
            linewidth=1.8,
            label=f"{class_name} AUC={auc:.4f}",
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="0.55", label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metadata: dict[str, Any],
    output_prefix: str | Path,
    fixed_fpr: float = 0.01,
    write_plots: bool = True,
    plot_path: str | Path | None = None,
) -> dict[str, Any]:
    class_names = list(metadata["class_names"])
    metrics, roc = compute_metrics(y_true, probabilities, class_names, fixed_fpr=fixed_fpr)
    output_prefix = resolve(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_probabilities(probabilities)
    predictions = normalized.argmax(axis=1)
    np.save(output_prefix.with_name(output_prefix.name + "_probabilities.npy"), normalized)
    np.save(output_prefix.with_name(output_prefix.name + "_predictions.npy"), predictions)
    np.savez_compressed(output_prefix.with_name(output_prefix.name + "_roc.npz"), **roc)
    if write_plots:
        plot_roc_curves(roc, metrics["per_class_auc"], plot_path or output_prefix.with_name(output_prefix.name + "_roc.png"))
    write_json(output_prefix.with_name(output_prefix.name + "_metrics.json"), metrics)
    return metrics


def load_pickled_model(path: str | Path) -> Any:
    with resolve(path).open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a tree model with the shared benchmark evaluator.")
    parser.add_argument("--config", required=True, help="YAML/JSON config path.")
    parser.add_argument("--model", help="Override model path from the config.")
    parser.add_argument("--no-plots", action="store_true", help="Skip ROC PNG generation.")
    args = parser.parse_args()

    config = load_config(args.config)
    arrays = load_dataset(config)
    model_path = args.model or config["model"]["output_model_path"]
    model = load_pickled_model(model_path)
    probabilities = predict_probabilities(model, arrays["X_test"])
    metrics = evaluate_probabilities(
        arrays["y_test"],
        probabilities,
        arrays["metadata"],
        config["evaluation"]["output_prefix"],
        fixed_fpr=float(config["evaluation"].get("fixed_fpr", 0.01)),
        write_plots=not args.no_plots,
        plot_path=config["evaluation"].get("roc_plot_path"),
    )
    print(f"accuracy={metrics['accuracy']:.8f} macro_auc={metrics['macro_auc']:.8f}")


if __name__ == "__main__":
    main()
