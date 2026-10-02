# 📊 Student Performance Analysis

A data science mini-project analyzing factors that influence student exam performance, built with Python, statistical hypothesis testing, and an interactive Streamlit dashboard.

🔗 **[Live Demo](#)** *(add your deployed Streamlit link here once deployed)*

---

## 📌 Overview

This project applies a complete data science workflow — data cleaning, exploratory data analysis, statistical hypothesis testing, and predictive modeling — to a real-world dataset of 6,607 students, to identify which factors most significantly affect exam scores.


## ✨ Features

- **Data Cleaning Pipeline** — handles missing values, invalid entries, and outliers using IQR detection and median/mode imputation
- **Statistical Analysis** — descriptive statistics and three hypothesis tests (one-sample t-test, two-sample t-test, Pearson correlation test)
- **Interactive Visualizations** — distribution plots, boxplots, scatter plots, correlation heatmaps, and category comparisons
- **Predictive Model** — Linear Regression (R² ≈ 0.62) with a live "predict your score" tool
- **Streamlit Dashboard** — all of the above in one interactive web app

## 📂 Dataset

[Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) (Kaggle) — 6,607 records, 20 features including study hours, attendance, sleep, parental involvement, and more. Target variable: `Exam_Score`.

## 🔑 Key Findings

| Finding | Detail |
|---|---|
| Strongest predictor | Attendance (r = 0.58, p < 0.001) |
| Second strongest | Hours Studied (r = 0.43, p < 0.001) |
| Internet access effect | Statistically significant (p < 0.001), though small in magnitude |
| Model performance | R² = 0.62, MAE = 1.31 marks |

## 🛠️ Tech Stack

Python · Pandas · NumPy · SciPy · Matplotlib · Seaborn · scikit-learn · Streamlit


## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/Priya7025/student-performance-analysis.git
cd student-performance-analysis

# Set up virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## 📊 Methodology

1. **Data Preparation** — Invalid values (e.g., `Exam_Score` > 100) converted to NaN; outliers in `Hours_Studied` detected via IQR method; missing values imputed using median (numeric) / mode (categorical)
2. **Exploratory Data Analysis** — Mean, median, std dev, skewness computed for all numeric features; Pearson correlation matrix generated
3. **Hypothesis Testing** — One-sample t-test (benchmark comparison), two-sample t-test (group comparison), Pearson significance test — all at α = 0.05
4. **Visualization** — Seaborn/Matplotlib used for distribution, outlier, relationship, and category-comparison charts
5. **Predictive Modeling** — Linear Regression trained on an 80/20 train-test split, evaluated via MAE, RMSE, and R²

## 📄 License

This project is for academic purposes as part of a college mini-project submission.