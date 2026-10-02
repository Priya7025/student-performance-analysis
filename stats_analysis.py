import pandas as pd
from scipy import stats


def descriptive_stats(df, numeric_cols):
    """Return a DataFrame of mean, median, std dev, skewness for given columns."""
    summary = {}
    for col in numeric_cols:
        summary[col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std_dev": df[col].std(),
            "skewness": df[col].skew(),
        }
    return pd.DataFrame(summary).T


def test_one_sample(df, column, benchmark):
    """One-sample t-test: is the column's mean significantly different from benchmark?"""
    t_stat, p_val = stats.ttest_1samp(df[column], benchmark)
    print(f"Mean = {df[column].mean():.2f}")
    print(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.5f}")
    if p_val < 0.05:
        print("Reject H0 — significant difference from benchmark.")
    else:
        print("Fail to reject H0 — no significant difference.")
    return t_stat, p_val


def test_two_sample(df, column, group_col, group1_val, group2_val):
    """Two-sample t-test: do two groups differ significantly in `column`?"""
    group1 = df[df[group_col] == group1_val][column]
    group2 = df[df[group_col] == group2_val][column]

    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

    print(f"Mean ({group_col}={group1_val}) = {group1.mean():.2f}  (n={len(group1)})")
    print(f"Mean ({group_col}={group2_val}) = {group2.mean():.2f}  (n={len(group2)})")
    print(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.5f}")
    if p_val < 0.05:
        print("Reject H0 — significant difference between groups.")
    else:
        print("Fail to reject H0 — no significant difference.")
    return t_stat, p_val


def test_correlation(df, col1, col2):
    """Pearson correlation test between two numeric columns."""
    r_val, p_val = stats.pearsonr(df[col1], df[col2])
    print(f"Pearson r = {r_val:.3f}, p-value = {p_val:.6f}")
    if p_val < 0.05:
        print("Reject H0 — significant correlation exists.")
    else:
        print("Fail to reject H0 — no significant correlation.")
    return r_val, p_val


if __name__ == "__main__":
    df = pd.read_csv("data/StudentPerformance_clean.csv")

    numeric_cols = ["Hours_Studied", "Attendance", "Sleep_Hours",
                     "Previous_Scores", "Tutoring_Sessions",
                     "Physical_Activity", "Exam_Score"]

    print("DESCRIPTIVE STATISTICS")
    print(descriptive_stats(df, numeric_cols))

    print("\nCORRELATION WITH EXAM_SCORE")
    correlations = df[numeric_cols].corr()["Exam_Score"].sort_values(ascending=False)
    print(correlations)

    print("\n[TEST 1] One-sample t-test (benchmark = 70)")
    test_one_sample(df, "Exam_Score", 70)

    print("\n[TEST 2] Two-sample t-test (Internet_Access: Yes vs No)")
    test_two_sample(df, "Exam_Score", "Internet_Access", "Yes", "No")

    print("\n[TEST 3] Pearson correlation (Attendance vs Exam_Score)")
    test_correlation(df, "Attendance", "Exam_Score")