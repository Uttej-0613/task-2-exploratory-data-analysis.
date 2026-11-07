"""Reusable exploratory-data-analysis functions for the Titanic dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Pclass"]

def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    result = numeric.describe().T
    result["median"] = numeric.median()
    result["missing"] = numeric.isna().sum()
    result["skewness"] = numeric.skew()
    return result

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    count = df.isna().sum()
    return pd.DataFrame({
        "missing_count": count,
        "missing_percent": (count / len(df) * 100).round(2)
    }).sort_values("missing_percent", ascending=False)

def outlier_report_iqr(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    columns = columns or [c for c in NUMERIC_FEATURES if c in df.columns]
    rows = []
    for col in columns:
        values = df[col].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        rows.append({
            "feature": col, "q1": q1, "q3": q3, "iqr": iqr,
            "lower_fence": lower, "upper_fence": upper,
            "potential_outliers": count,
            "outlier_percent": round(count / len(values) * 100, 2)
        })
    return pd.DataFrame(rows)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["FamilySize"] = result["SibSp"].fillna(0) + result["Parch"].fillna(0) + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
    result["AgeGroup"] = pd.cut(
        result["Age"], bins=[0, 12, 18, 35, 60, np.inf],
        labels=["Child", "Teen", "Young adult", "Adult", "Senior"],
        include_lowest=True
    )
    return result

def create_visualizations(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved = []
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.flat, numeric):
        sns.histplot(data=df, x=col, kde=True, ax=ax, color="#3977a8")
        ax.set_title(f"Distribution of {col}")
    for ax in axes.flat[len(numeric):]:
        ax.remove()
    fig.tight_layout()
    path = output / "numeric_histograms.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, len(numeric), figsize=(4 * len(numeric), 5))
    if len(numeric) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric):
        sns.boxplot(data=df, y=col, ax=ax, color="#9bc2d8")
        ax.set_title(f"Boxplot: {col}")
    fig.tight_layout()
    path = output / "numeric_boxplots.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (numeric features)")
    fig.tight_layout()
    path = output / "correlation_heatmap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    if {"Sex", "Survived"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=df, x="Sex", y="Survived", errorbar=None, palette="deep", hue="Sex", legend=False, ax=ax)
        ax.set_ylabel("Survival rate")
        ax.set_title("Survival rate by sex")
        fig.tight_layout()
        path = output / "survival_by_sex.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    if {"Pclass", "Survived"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=df, x="Pclass", y="Survived", errorbar=None, palette="deep", hue="Pclass", legend=False, ax=ax)
        ax.set_ylabel("Survival rate")
        ax.set_title("Survival rate by passenger class")
        fig.tight_layout()
        path = output / "survival_by_class.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    if {"Age", "Fare", "Survived"}.issubset(df.columns):
        fig = px.scatter(
            df, x="Age", y="Fare", color="Survived",
            hover_data=["Pclass", "Sex"], title="Age vs Fare, colored by survival"
        )
        path = output / "age_vs_fare.html"
        fig.write_html(path)
        saved.append(path)
    return saved

def generate_report(df: pd.DataFrame, output_path: str | Path) -> Path:
    report = Path(output_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    missing = missing_value_report(df)
    stats = summary_statistics(df)
    outliers = outlier_report_iqr(df)
    survival = df.groupby("Sex", observed=False)["Survived"].mean().sort_values(ascending=False)
    class_survival = df.groupby("Pclass", observed=False)["Survived"].mean().sort_index()
    lines = [
        "# Titanic Dataset — EDA Results", "",
        f"- Rows: **{len(df):,}**",
        f"- Columns: **{df.shape[1]}**",
        f"- Duplicate rows: **{int(df.duplicated().sum())}**", "",
        "## Summary statistics", "", stats.round(3).to_markdown(), "",
        "## Missing values", "", missing.to_markdown(), "",
        "## Potential outliers (1.5 × IQR rule)", "",
        outliers.round(3).to_markdown(index=False), "",
        "## Survival-rate comparisons", "", "### By sex", "",
        survival.rename("survival_rate").to_frame().to_markdown(), "",
        "### By passenger class", "",
        class_survival.rename("survival_rate").to_frame().to_markdown(), "",
        "## Interpretation notes", "",
        "- These are descriptive associations; they do not establish causation.",
        "- IQR flags are potential outliers, not automatically erroneous observations.",
        "- Missing values should be handled based on the downstream modeling objective.",
        "- Correlation only measures linear association for numeric columns.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
