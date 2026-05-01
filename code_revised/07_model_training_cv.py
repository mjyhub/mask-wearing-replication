import pickle
import runpy
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


cfg = runpy.run_path(Path(__file__).with_name("01_config.py"))

UPSAMPLE_MODELS = {"model_1a", "model_1b", "model_2a", "model_2b"}


def load_dataset(model_name):
    X_train = pd.read_csv(cfg["DATA_DIR"] / f"X_train_{model_name}.csv", keep_default_na=False)
    X_test = pd.read_csv(cfg["DATA_DIR"] / f"X_test_{model_name}.csv", keep_default_na=False)
    y_train = pd.read_csv(cfg["DATA_DIR"] / f"y_train_{model_name}.csv").values.ravel()
    y_test = pd.read_csv(cfg["DATA_DIR"] / f"y_test_{model_name}.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def build_model(model_type):
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=5000, random_state=cfg["MODEL_SEED"])
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            random_state=cfg["MODEL_SEED"],
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def maybe_upsample(X, y, model_name):
    if model_name not in UPSAMPLE_MODELS:
        return X, y
    sampler = RandomOverSampler(random_state=cfg["MODEL_SEED"])
    return sampler.fit_resample(X, y)


def metrics(y_true, preds, scores):
    return {
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, scores),
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds),
    }


def cross_validate_model(model_name, model_type, X, y):
    splitter = StratifiedShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=cfg["MODEL_SEED"],
    )
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y[valid_idx]

        X_fit, y_fit = maybe_upsample(X_train, y_train, model_name)
        model = build_model(model_type)
        model.fit(X_fit, y_fit)

        preds = model.predict(X_valid)
        scores = model.predict_proba(X_valid)[:, 1]
        row = {"model": model_name, "model_type": model_type, "fold": fold}
        row.update(metrics(y_valid, preds, scores))
        rows.append(row)
    return rows


def fit_final_model(model_name, model_type, X_train, y_train, X_test, y_test):
    X_fit, y_fit = maybe_upsample(X_train, y_train, model_name)
    model = build_model(model_type)
    model.fit(X_fit, y_fit)

    with open(cfg["MODELS_DIR"] / f"{model_name}_{model_type}.pkl", "wb") as handle:
        pickle.dump(model, handle)

    preds = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    row = {
        "model": model_name,
        "model_type": model_type,
        "upsampled_training": model_name in UPSAMPLE_MODELS,
    }
    row.update(metrics(y_test, preds, scores))
    return row


def main():
    cfg["ensure_output_dirs"]()

    cv_rows = []
    test_rows = []
    model_types = ["logistic_regression", "random_forest"]

    for model_name in cfg["MODEL_SPECS"]:
        X_train, X_test, y_train, y_test = load_dataset(model_name)
        for model_type in model_types:
            cv_rows.extend(cross_validate_model(model_name, model_type, X_train, y_train))
            test_rows.append(fit_final_model(model_name, model_type, X_train, y_train, X_test, y_test))

    cv_results = pd.DataFrame(cv_rows)
    test_results = pd.DataFrame(test_rows)
    cv_results.to_csv(cfg["RESULTS_DIR"] / "cross_validation_metrics.csv", index=False)
    test_results.to_csv(cfg["RESULTS_DIR"] / "final_test_metrics.csv", index=False)

    print("Cross-validation means:")
    print(
        cv_results.groupby(["model", "model_type"])[
            ["precision", "recall", "roc_auc", "accuracy", "f1"]
        ]
        .mean()
        .round(3)
        .reset_index()
        .to_string(index=False)
    )
    print("\nHeld-out test metrics:")
    print(test_results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

