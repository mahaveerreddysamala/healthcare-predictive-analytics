"""Batch prediction entry point."""
from __future__ import annotations

import argparse
import joblib
import pandas as pd


def main(model_path: str, input_path: str, output_path: str) -> None:
    model = joblib.load(model_path)
    data = pd.read_csv(input_path)
    data["readmission_risk"] = model.predict_proba(data)[:, 1]
    data["risk_segment"] = pd.cut(data["readmission_risk"], [-1, .33, .66, 1], labels=["Low", "Medium", "High"])
    data.to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    main(args.model, args.input, args.output)
