import pandas as pd
import pytest
from src.data_prep import clean_data, split_data


def test_clean_data_normalizes_and_deduplicates():
    df = pd.DataFrame({"Readmitted": [1, 1, 0, 0], "Patient ID": [10, 10, 11, 12]})
    result = clean_data(df)
    assert list(result.columns) == ["readmitted", "patient_id"]
    assert len(result) == 3


def test_clean_data_requires_target():
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame({"age": [50]}))


def test_split_data_preserves_target():
    df = pd.DataFrame({"readmitted": [0, 1] * 10, "age": list(range(40, 60))})
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert "readmitted" not in X_train.columns
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
