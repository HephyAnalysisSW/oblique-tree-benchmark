from __future__ import annotations

import argparse
import pickle

from xgboost import XGBClassifier

from data_utils import load_config, load_dataset, resolve, set_seed, write_json
from evaluate import evaluate_probabilities, predict_probabilities


def build_classifier(config: dict) -> XGBClassifier:
    options = dict(config["model"]["xgboost"])
    return XGBClassifier(**options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reference XGBoost BDT on the fixed jet benchmark split.")
    parser.add_argument("--config", default="configs/bdt_config.yaml", help="YAML config path.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    arrays = load_dataset(config)
    model = build_classifier(config)
    model.fit(
        X=arrays["X_train"],
        y=arrays["y_train"],
        eval_set=[
            (arrays["X_train"], arrays["y_train"]),
            (arrays["X_val"], arrays["y_val"]),
        ],
        verbose=True,
    )

    model_path = resolve(config["model"]["output_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "config": {key: value for key, value in config.items() if key != "_config_path"},
        "metadata": arrays["metadata"],
        "history": model.evals_result() if hasattr(model, "evals_result") else {},
    }
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(model_path.with_suffix(".history.json"), payload["history"])

    probabilities = predict_probabilities(model, arrays["X_test"])
    metrics = evaluate_probabilities(
        arrays["y_test"],
        probabilities,
        arrays["metadata"],
        config["evaluation"]["output_prefix"],
        fixed_fpr=float(config["evaluation"].get("fixed_fpr", 0.01)),
    )
    print(f"saved_model={model_path}")
    print(f"accuracy={metrics['accuracy']:.8f} macro_auc={metrics['macro_auc']:.8f}")


if __name__ == "__main__":
    main()
