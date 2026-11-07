from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_data import download_dataset
from src.eda import (
    load_data, summary_statistics, missing_value_report,
    outlier_report_iqr, add_engineered_features,
    create_visualizations, generate_report
)

def main():
    data_path = download_dataset()
    df = load_data(data_path)
    output = ROOT / "outputs"
    reports = ROOT / "reports"
    output.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    summary_statistics(df).to_csv(output / "summary_statistics.csv")
    missing_value_report(df).to_csv(output / "missing_values.csv")
    outlier_report_iqr(df).to_csv(output / "outlier_report.csv", index=False)
    add_engineered_features(df).to_csv(output / "titanic_with_features.csv", index=False)
    charts = create_visualizations(df, output)
    report = generate_report(df, reports / "EDA_REPORT.md")
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Generated {len(charts)} visualizations in {output}")
    print(f"Generated report: {report}")

if __name__ == "__main__":
    main()
