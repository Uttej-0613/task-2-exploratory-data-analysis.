import pandas as pd
from src.eda import summary_statistics, missing_value_report, outlier_report_iqr, add_engineered_features

def sample_data():
    return pd.DataFrame({
        "Survived": [0, 1, 1, 0],
        "Pclass": [3, 1, 2, 3],
        "Sex": ["male", "female", "female", "male"],
        "Age": [22.0, 38.0, None, 35.0],
        "SibSp": [1, 1, 0, 0],
        "Parch": [0, 0, 0, 0],
        "Fare": [7.25, 71.28, 13.0, 8.05],
    })

def test_summary_includes_median_and_missing():
    result = summary_statistics(sample_data())
    assert "median" in result.columns
    assert result.loc["Age", "missing"] == 1

def test_missing_value_percent():
    result = missing_value_report(sample_data())
    assert result.loc["Age", "missing_count"] == 1
    assert result.loc["Age", "missing_percent"] == 25.0

def test_iqr_report_has_expected_columns():
    result = outlier_report_iqr(sample_data())
    assert {"feature", "iqr", "potential_outliers"}.issubset(result.columns)

def test_engineered_features():
    result = add_engineered_features(sample_data())
    assert result.loc[0, "FamilySize"] == 2
    assert result.loc[2, "IsAlone"] == 1
