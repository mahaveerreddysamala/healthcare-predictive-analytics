"""Train a reusable readmission-risk classification pipeline."""
from __future__ import annotations

import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import clean_data, split_data


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocessor", prep), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def main(path: str, output: str) -> None:
    df = clean_data(pd.read_csv(path))
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, probability):.4f}")
    print(classification_report(y_test, model.predict(X_test)))
    joblib.dump(model, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/readmission_model.joblib")
    args = parser.parse_args()
    main(args.data, args.output)
