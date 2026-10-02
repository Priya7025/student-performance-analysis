import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_distribution(df, column):
    """Histogram + KDE for a single numeric column."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df[column], kde=True, color="steelblue", bins=20, ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Number of Students")
    fig.tight_layout()
    return fig


def plot_boxplots(df, numeric_cols):
    """Boxplots for multiple numeric columns side by side."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df[numeric_cols], ax=ax)
    ax.set_title("Boxplots of Numeric Variables")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_scatter(df, x_col, y_col):
    """Scatter plot with trend line between two numeric columns."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=df, x=x_col, y=y_col,
                scatter_kws={"alpha": 0.4, "color": "teal"},
                line_kws={"color": "red"}, ax=ax)
    ax.set_title(f"{x_col} vs {y_col}")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df, numeric_cols):
    """Correlation heatmap for numeric columns."""
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_bar_by_category(df, cat_col, value_col):
    """Average value_col grouped by a categorical column."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df, x=cat_col, y=value_col, errorbar="sd", ax=ax)
    ax.set_title(f"Average {value_col} by {cat_col}")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = pd.read_csv("data/StudentPerformance_clean.csv")

    numeric_cols = ["Hours_Studied", "Attendance", "Sleep_Hours",
                     "Previous_Scores", "Tutoring_Sessions",
                     "Physical_Activity", "Exam_Score"]

    # Generate and save each chart as a PNG (used in your Word report)
    fig1 = plot_distribution(df, "Exam_Score")
    fig1.savefig("fig1_exam_score_distribution.png")

    fig2 = plot_boxplots(df, numeric_cols)
    fig2.savefig("fig2_boxplots.png")

    fig3 = plot_scatter(df, "Attendance", "Exam_Score")
    fig3.savefig("fig3_attendance_vs_score.png")

    fig4 = plot_correlation_heatmap(df, numeric_cols)
    fig4.savefig("fig4_correlation_heatmap.png")

    fig5 = plot_bar_by_category(df, "Parental_Education_Level", "Exam_Score")
    fig5.savefig("fig5_parental_education_bar.png")

    print("All charts saved successfully.")