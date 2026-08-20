# Architecture

```text
CSV / Warehouse
      |
      v
Data validation + cleaning
      |
      v
Feature preprocessing
      |
      v
Classification model
      |
      +--> Evaluation (ROC-AUC / Precision / Recall / F1)
      |
      v
Persisted model
      |
      v
Batch risk scoring
```

The pipeline separates data preparation, model training, and inference so each stage can be tested and replaced independently. No real patient data or credentials belong in the repository.
