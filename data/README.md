# Data

No protected or real patient information is stored in this repository.

To run the training pipeline, provide a local CSV with a binary `readmitted` target and any numeric/categorical predictor columns.

Example command:

```bash
python src/train.py --data data/patients.csv --output models/readmission_model.joblib
```

For a portfolio repository, synthetic or de-identified data should be used. Never commit PHI, credentials, access tokens, or other sensitive information.
