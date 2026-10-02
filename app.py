import streamlit as st
import pandas as pd

import data_processing
import stats_analysis
import visualization
import model

# ---- Page configuration (must be the first Streamlit command) ----
st.set_page_config(page_title="Student Performance Analysis", layout="wide")


# ---- Cached data loading ----
@st.cache_data
def get_clean_data():
    return pd.read_csv("data/StudentPerformance_clean.csv")


df = get_clean_data()

FEATURE_COLS = ["Hours_Studied", "Attendance", "Sleep_Hours",
                 "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]
TARGET_COL = "Exam_Score"


# ---- Cached model training ----
@st.cache_resource
def get_trained_model():
    X, y = model.prepare_features(df, FEATURE_COLS, TARGET_COL)
    model_obj = model.train_model(X, y)
    return model_obj


trained_model = get_trained_model()

# ---- Title ----
st.title("📊 Student Performance Analysis")
st.markdown("Mini Project — Python for Data Science (BE05000231)")

with st.expander("🔍 View raw dataset preview"):
    st.dataframe(df.head(20))

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 EDA & Statistics", "📊 Visualizations", "🤖 Predict Score", "ℹ️ About"
])

# =========================================================
# TAB 1: EDA & Statistics
# =========================================================
with tab1:
    st.header("Descriptive Statistics")

    numeric_cols = ["Hours_Studied", "Attendance", "Sleep_Hours",
                     "Previous_Scores", "Tutoring_Sessions", "Physical_Activity", "Exam_Score"]

    stats_table = stats_analysis.descriptive_stats(df, numeric_cols)
    st.dataframe(stats_table)

    st.header("Correlation with Exam Score")
    correlations = df[numeric_cols].corr()["Exam_Score"].sort_values(ascending=False)
    st.bar_chart(correlations.drop("Exam_Score"))

    st.header("Hypothesis Testing")

    st.subheader("Test 1: One-sample t-test (benchmark = 70)")
    t_stat, p_val = stats_analysis.test_one_sample(df, "Exam_Score", 70)
    st.write(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.5f}")
    if p_val < 0.05:
        st.success("Reject H0 — significant difference from benchmark.")
    else:
        st.info("Fail to reject H0 — no significant difference.")

    st.subheader("Test 2: Internet Access (Yes vs No)")
    t_stat2, p_val2 = stats_analysis.test_two_sample(
        df, "Exam_Score", "Internet_Access", "Yes", "No"
    )
    st.write(f"t-statistic = {t_stat2:.3f}, p-value = {p_val2:.5f}")
    if p_val2 < 0.05:
        st.success("Reject H0 — significant difference between groups.")
    else:
        st.info("Fail to reject H0 — no significant difference.")

# =========================================================
# TAB 2: Visualizations
# =========================================================
with tab2:
    st.header("Exploratory Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Exam Score Distribution")
        fig1 = visualization.plot_distribution(df, "Exam_Score")
        st.pyplot(fig1)

    with col2:
        st.subheader("Boxplots (Outlier Check)")
        fig2 = visualization.plot_boxplots(df, numeric_cols)
        st.pyplot(fig2)

    st.subheader("Attendance vs Exam Score")
    fig3 = visualization.plot_scatter(df, "Attendance", "Exam_Score")
    st.pyplot(fig3)

    st.subheader("Correlation Heatmap")
    fig4 = visualization.plot_correlation_heatmap(df, numeric_cols)
    st.pyplot(fig4)

    st.subheader("Average Exam Score by Parental Education")
    fig5 = visualization.plot_bar_by_category(df, "Parental_Education_Level", "Exam_Score")
    st.pyplot(fig5)

# =========================================================
# TAB 3: Predict Score
# =========================================================
with tab3:
    st.header("Predict a Student's Exam Score")
    st.write("Adjust the sliders below to simulate a student's profile.")

    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.slider("Hours Studied (per week)", 0, 44, 20)
        attendance = st.slider("Attendance (%)", 60, 100, 80)
        sleep_hours = st.slider("Sleep Hours", 4, 10, 7)

    with col2:
        previous_scores = st.slider("Previous Scores", 50, 100, 75)
        tutoring_sessions = st.slider("Tutoring Sessions (per month)", 0, 8, 1)
        physical_activity = st.slider("Physical Activity (hrs/week)", 0, 6, 3)

    if st.button("Predict Exam Score"):
        predicted = model.predict_new_student(
            trained_model, FEATURE_COLS,
            Hours_Studied=hours_studied,
            Attendance=attendance,
            Sleep_Hours=sleep_hours,
            Previous_Scores=previous_scores,
            Tutoring_Sessions=tutoring_sessions,
            Physical_Activity=physical_activity
        )
        st.success(f"Predicted Exam Score: **{predicted:.2f}**")
        st.caption("Note: Linear Regression model trained on the full cleaned dataset "
                    "(R² ≈ 0.62). Prediction is an estimate, not a guarantee.")

# =========================================================
# TAB 4: About
# =========================================================
with tab4:
    st.header("About This Project")

    st.markdown("""
    **Project:** Student Performance Analysis using Descriptive and Inferential Statistics

    **Course:** Python for Data Science (BE05000231) — B.E. Semester V

    **Dataset:** [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
    (Kaggle) — 6,607 student records, 20 features.

    **Workflow:**
    1. **Data Preparation** — cleaned invalid values, detected outliers (IQR method),
       imputed missing values (median for numeric, mode for categorical).
    2. **Exploratory Data Analysis** — descriptive statistics, correlation analysis.
    3. **Hypothesis Testing** — one-sample t-test, two-sample t-test, Pearson
       correlation significance test.
    4. **Visualization** — histograms, boxplots, scatter plots, heatmaps, bar charts.
    5. **Predictive Modeling (bonus)** — Linear Regression, trained/tested with an
       80/20 split, R² ≈ 0.62.

    **Tech stack:** Python, Pandas, NumPy, SciPy, Matplotlib, Seaborn, scikit-learn, Streamlit.

    **Key Finding:** Attendance (r = 0.58) and Hours Studied (r = 0.43) are the
    strongest statistically significant predictors of Exam Score in this dataset.
    """)

    st.divider()
    st.caption("Built as part of a mini project submission. Source code on GitHub.")