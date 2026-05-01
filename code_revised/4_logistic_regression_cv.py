#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logistic regression cross-validation for all prediction tasks.

This script keeps the same output file names and metric keys as the original
replication workflow, so the downstream result collection scripts can be used
without changing their expected inputs.
"""

import pickle
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    f1_score
)
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate


MODEL_NUMBERS = [
    "model_1",
    "model_1a",
    "model_1b",
    "model_2",
    "model_2a",
    "model_2b"
]

UPSAMPLE_MODELS = {
    "model_1a",
    "model_1b",
    "model_2a",
    "model_2b"
}

METRIC_LIST = [
    "precision",
    "recall",
    "roc_auc",
    "accuracy",
    "f1"
]

N_SPLITS = 5
SEED = 20240627


def load_training_data(model_number):
    X = pd.read_csv(
        f"../data/X_train_{model_number}.csv",
        keep_default_na=False
    )

    y = pd.read_csv(
        f"../data/y_train_{model_number}.csv",
        keep_default_na=False
    ).values.ravel()

    return X, y


def make_cv_splitter():
    return StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=1 / N_SPLITS,
        random_state=SEED
    )


def make_logistic_model():
    return LogisticRegression(
        max_iter=5000
    )


def run_standard_cv(X, y):
    model = make_logistic_model()
    cv = make_cv_splitter()

    return cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=METRIC_LIST,
        return_train_score=False
    )


def run_upsampled_cv(X, y):
    cv = make_cv_splitter()

    scores = {
        "fold": [],
        "test_precision": [],
        "test_recall": [],
        "test_roc_auc": [],
        "test_accuracy": [],
        "test_f1": []
    }

    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]

        X_valid = X.iloc[valid_idx]
        y_valid = y[valid_idx]

        upsampler = RandomOverSampler(random_state=SEED + fold_id)
        X_train_up, y_train_up = upsampler.fit_resample(X_train, y_train)

        model = make_logistic_model()
        model.fit(X_train_up, y_train_up)

        preds = model.predict(X_valid)
        probs = model.predict_proba(X_valid)[:, 1]

        scores["fold"].append(fold_id)
        scores["test_precision"].append(precision_score(y_valid, preds))
        scores["test_recall"].append(recall_score(y_valid, preds))
        scores["test_roc_auc"].append(roc_auc_score(y_valid, probs))
        scores["test_accuracy"].append(accuracy_score(y_valid, preds))
        scores["test_f1"].append(f1_score(y_valid, preds))

    for key in [
        "test_precision",
        "test_recall",
        "test_roc_auc",
        "test_accuracy",
        "test_f1"
    ]:
        scores[key] = np.array(scores[key])

    return scores


def save_cv_scores(model_number, cv_scores):
    output_path = f"../results/{model_number}_logistic_reg.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(cv_scores, f)

    return output_path


def summarize_scores(model_number, cv_scores):
    print(model_number)
    print("Mean recall:", round(cv_scores["test_recall"].mean(), 3))
    print("Mean roc:", round(cv_scores["test_roc_auc"].mean(), 3))
    print("Mean accuracy:", round(cv_scores["test_accuracy"].mean(), 3))


def run_one_model(model_number):
    X, y = load_training_data(model_number)

    if model_number in UPSAMPLE_MODELS:
        cv_scores = run_upsampled_cv(X, y)
    else:
        cv_scores = run_standard_cv(X, y)

    output_path = save_cv_scores(model_number, cv_scores)
    summarize_scores(model_number, cv_scores)

    print(f"Saved to {output_path}")
    print("-" * 60)


def main():
    for model_number in MODEL_NUMBERS:
        run_one_model(model_number)


if __name__ == "__main__":
    main()

