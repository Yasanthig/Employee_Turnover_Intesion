"""Train and evaluate the new Categorical Naive Bayes model.

Install requirements with: python -m pip install -r requirements.txt
Run: python turnover_nb_project.py --data "path/to/research_data.csv"
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OrdinalEncoder


TARGET = "Do you have any turnover intention?"
TARGET_LABEL = "Stated turnover intention"
DEFAULT_DATA = Path(__file__).resolve().parent / "turnover_data.csv"
SEED = 42


def shift_category_codes(values: np.ndarray) -> np.ndarray:
    """Reserve 0 for unseen values and shift observed category codes to 1+."""
    values = np.asarray(values, dtype=np.int64)
    return np.where(values < 0, 0, values + 1)


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "bin",
                KBinsDiscretizer(
                    n_bins=5, encode="ordinal", strategy="quantile", subsample=None
                ),
            ),
            ("category_codes", FunctionTransformer(shift_category_codes)),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("category_codes", FunctionTransformer(shift_category_codes)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_steps, numeric),
            ("categorical", categorical_steps, categorical),
        ],
        remainder="drop",
    )
    return Pipeline(
        [("preprocess", preprocessing), ("model", CategoricalNB(alpha=1.0))]
    )


def resolve_data_path(requested: Path) -> Path:
    """Resolve a relative CSV path and fall back clearly to the bundled dataset."""
    candidate = requested
    if not candidate.is_absolute() and not candidate.is_file():
        candidate = Path(__file__).resolve().parent / candidate
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate

    if DEFAULT_DATA.is_file() and DEFAULT_DATA.stat().st_size > 0:
        print(
            f"Warning: '{requested}' is missing or empty. "
            f"Using the included dataset instead: {DEFAULT_DATA}"
        )
        return DEFAULT_DATA
    raise FileNotFoundError(
        f"CSV not found or empty: {requested}. Put a non-empty CSV in the project "
        "folder or run with --data <full-path-to-csv>."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("turnover_results"))
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)
    raw = pd.read_csv(data_path)
    if TARGET not in raw.columns:
        raise ValueError(f"Target column not found: {TARGET}")
    if raw[TARGET].isna().any():
        raise ValueError("Target contains missing values; resolve these before training.")
    labels = set(raw[TARGET].astype(str).str.strip().unique())
    if labels != {"Yes", "No"}:
        raise ValueError(f"Expected target labels Yes/No, found: {sorted(labels)}")

    duplicate_count = int(raw.duplicated().sum())
    missing_cell_count = int(raw.isna().sum().sum())
    data = raw.drop_duplicates().reset_index(drop=True)
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(str).str.strip().map({"No": 0, "Yes": 1}).astype(int)
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    pipeline = build_pipeline(numeric, categorical)

    # Keep a stratified holdout untouched until the final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "f1_yes": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_result = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=1, error_score="raise", return_train_score=False,
        )
        evaluation_model = pipeline.fit(X_train, y_train)

    prediction = evaluation_model.predict(X_test)
    probability = evaluation_model.predict_proba(X_test)[:, 1]
    metrics = {
        "algorithm": "Categorical Naive Bayes",
        "rows_after_exact_duplicate_removal": int(len(data)),
        "predictor_count": int(X.shape[1]),
        "target": TARGET_LABEL,
        "source_target_column": TARGET,
        "class_counts": {"No": int((y == 0).sum()), "Yes": int((y == 1).sum())},
        "duplicates_removed": duplicate_count,
        "missing_cells_before_imputation": missing_cell_count,
        "split": "80/20 stratified holdout; random_state=42",
        "training_validation": "5-fold stratified cross-validation and a separate 20% held-out evaluation; only Categorical Naive Bayes is trained",
        "cv_f1_yes_mean": float(np.mean(cv_result["test_f1_yes"])),
        "cv_f1_yes_sd": float(np.std(cv_result["test_f1_yes"], ddof=1)),
        "cv_balanced_accuracy_mean": float(np.mean(cv_result["test_balanced_accuracy"])),
        "cv_roc_auc_mean": float(np.mean(cv_result["test_roc_auc"])),
        "test_rows": int(len(y_test)),
        "test_positive_count": int(y_test.sum()),
        "deployment_model_training_rows": int(len(X)),
        "deployment_model": "Refit on all labeled rows after holdout evaluation; used by the app for new responses.",
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_precision_yes": float(precision_score(y_test, prediction, zero_division=0)),
        "test_recall_yes": float(recall_score(y_test, prediction, zero_division=0)),
        "test_f1_yes": float(f1_score(y_test, prediction, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probability)),
        "test_pr_auc": float(average_precision_score(y_test, probability)),
        "test_confusion_matrix_no_yes": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        "test_classification_report": classification_report(
            y_test, prediction, labels=[0, 1], target_names=["No", "Yes"],
            output_dict=True, zero_division=0,
        ),
        "caution": (
            "Only 203 survey responses and 47 Yes labels; single holdout estimates are uncertain. "
            "Predicts stated intention, not actual turnover. Associations are not causal."
        ),
    }

    importance_result = permutation_importance(
        evaluation_model, X_test, y_test, scoring="f1", n_repeats=30,
        random_state=SEED, n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_f1_decrease": importance_result.importances_mean,
            "std_f1_decrease": importance_result.importances_std,
        }
    ).sort_values("mean_f1_decrease", ascending=False)

    args.output.mkdir(parents=True, exist_ok=True)
    # Refit after evaluation so the app's deployment artifact uses every
    # labeled response. The reported test metrics remain from evaluation_model.
    deployment_model = pipeline.fit(X, y)
    joblib.dump(deployment_model, args.output / "categorical_nb_model.joblib")
    importance.to_csv(args.output / "naive_bayes_permutation_importance.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Categorical Naive Bayes results")
    print(f"5-fold CV F1 for Yes: {metrics['cv_f1_yes_mean']:.3f} ± {metrics['cv_f1_yes_sd']:.3f}")
    print(classification_report(
        y_test, prediction, labels=[0, 1], target_names=["No", "Yes"], zero_division=0
    ))
    print("Top permutation-importance features:")
    print(importance.head(10).to_string(index=False))
    print(f"\nSaved results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
