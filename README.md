# Healthcare Predictive Analytics

An end-to-end machine learning project for predicting patient readmission risk and identifying operational factors associated with avoidable hospital utilization.

## Project Overview

This portfolio project demonstrates a production-oriented analytics workflow: data validation, exploratory analysis, feature engineering, model training, evaluation, explainability, and reusable prediction code.

## Business Objective

Healthcare organizations need to identify patients at elevated risk of readmission so care-management teams can prioritize follow-up resources. The project predicts a binary readmission outcome from demographic, utilization, and clinical proxy features.

## Tech Stack

- Python 3.11
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- SQL
- Jupyter
- Joblib

## Repository Structure

```text
healthcare-predictive-analytics/
├── data/
│   └── README.md
├── notebooks/
│   └── 01_exploratory_analysis.md
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── train.py
│   └── predict.py
├── sql/
│   └── patient_risk_analysis.sql
├── tests/
│   └── test_data_prep.py
├── requirements.txt
└── README.md
```

## Workflow

1. Validate and clean source data.
2. Explore readmission patterns and missingness.
3. Engineer utilization and demographic features.
4. Train a baseline logistic regression model and tree-based model.
5. Evaluate ROC-AUC, precision, recall, F1, and confusion matrix.
6. Select a model based on business cost and recall requirements.
7. Persist the trained model for batch scoring.

## Example Outcome

The project is designed to optimize for **recall of high-risk patients** rather than accuracy alone, because missing a genuinely high-risk patient can be more costly than generating an additional care-management review.

## Responsible AI

This project is educational and uses synthetic/demo data. Predictions should not be used for clinical decisions without appropriate clinical validation, governance, bias testing, privacy controls, and regulatory review.
