# Task 2 — Exploratory Data Analysis (EDA)

A reproducible EDA project using the Titanic passenger dataset. It covers descriptive statistics, histograms, boxplots, feature relationships, correlation analysis, outlier checks, and evidence-based observations.

## Dataset
- Titanic training dataset (expected 891 passenger records).
- Source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- The CSV is downloaded by the notebook/script on first run and is not committed.

## Run in Google Colab
1. Open notebooks/Task_2_EDA_Titanic.ipynb in Google Colab.
2. Select Runtime → Run all while connected to the internet.
3. The notebook installs dependencies, downloads the CSV, performs analysis and displays charts.

## Run locally
    pip install -r requirements.txt
    python scripts/run_eda.py

## Project contents
- notebooks/: end-to-end notebook
- src/eda.py: reusable EDA functions
- scripts/: dataset downloader and command-line runner
- tests/: automated unit tests
- reports/: report template and generated-report destination
- INTERVIEW_QUESTIONS.md: answers to assignment interview questions

## Analysis
Summary statistics, missing-value analysis, IQR outlier checks, feature engineering, histograms, boxplots, correlation heatmap, survival comparisons, and an interactive Plotly chart.

## Limitations
This is exploratory analysis, not causal inference or a predictive model. Run the notebook in Colab and inspect generated results before treating the analysis as complete. Correlation does not imply causation; IQR flags are potential outliers, not necessarily errors.
