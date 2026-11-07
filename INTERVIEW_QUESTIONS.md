# Task 2 — EDA Interview Questions

### 1. What is the purpose of EDA?
Exploratory Data Analysis is used to understand a dataset before modeling. It helps reveal data types, distributions, missing values, outliers, relationships, unusual patterns and possible data-quality problems.

### 2. How do boxplots help in understanding a dataset?
A boxplot displays the median, lower and upper quartiles, interquartile range and potential outliers. It makes spread, skew and extreme observations easier to compare across groups. Points outside the whiskers are candidates for investigation, not automatically errors.

### 3. What is correlation and why is it useful?
Correlation measures the strength and direction of association between two variables. Pearson correlation measures linear association and ranges from -1 to +1. It helps identify related features, but correlation does not prove causation and may miss nonlinear relationships.

### 4. How do you detect skewness in data?
Use a histogram or density plot to inspect asymmetry and calculate a skewness statistic. Positive skew has a longer right tail; negative skew has a longer left tail. Compare mean and median as an additional clue. A value near zero suggests less asymmetry, but does not guarantee a normal distribution.

### 5. What is multicollinearity?
Multicollinearity occurs when two or more predictor variables are strongly related. It can make regression coefficients unstable and harder to interpret. Investigate it using correlation matrices and Variance Inflation Factor (VIF).

### 6. What tools do you use for EDA?
Pandas and NumPy for data handling and statistics; Matplotlib and Seaborn for static plots; Plotly for interactive visualizations; Jupyter Notebook or Google Colab for an interactive workflow.

### 7. Can you explain a time when EDA helped you find a problem?
In this project, the missing-value report identifies incomplete columns such as Age, Cabin and Embarked. Boxplots and the IQR table flag unusually high fares or ages for review. This informs whether to investigate, impute, transform or retain those values. Confirm the findings by running the notebook and inspecting outputs.

### 8. What is the role of visualization in ML?
Visualization helps detect data-quality issues, understand feature distributions, spot outliers, examine relationships and communicate findings. It supports feature engineering and model selection, but does not replace statistical testing or validation.
