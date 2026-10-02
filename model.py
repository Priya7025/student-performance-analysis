import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_features(df, feature_cols, target_col):
    """Select numeric features (X) and target (y) for modeling."""
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def train_model(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on unseen test data."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  (Mean Absolute Error)       : {mae:.3f}")
    print(f"RMSE (Root Mean Squared Error)    : {rmse:.3f}")
    print(f"R² Score (variance explained)     : {r2:.3f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def show_coefficients(model, feature_cols):
    """Print each feature's coefficient — its effect on Exam_Score."""
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"{feat:20s}: {coef:+.4f}")
    print(f"{'Intercept':20s}: {model.intercept_:+.4f}")

def predict_new_student(model, feature_cols, **kwargs):
    """Predict Exam_Score for a new student given feature values as keyword args."""
    input_df = pd.DataFrame([kwargs], columns=feature_cols)
    prediction = model.predict(input_df)[0]
    return prediction


if __name__ == "__main__":
    df = pd.read_csv("data/StudentPerformance_clean.csv")

    feature_cols = ["Hours_Studied", "Attendance", "Sleep_Hours",
                     "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]
    target_col = "Exam_Score"

    X, y = prepare_features(df, feature_cols, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    model = train_model(X_train, y_train)

    print("MODEL COEFFICIENTS")
    show_coefficients(model, feature_cols)

    print("\nMODEL EVALUATION (on unseen test data)")
    evaluate_model(model, X_test, y_test)

    print("\nSAMPLE PREDICTION")
    sample_score = predict_new_student(
        model, feature_cols,
        Hours_Studied=25, Attendance=90, Sleep_Hours=7,
        Previous_Scores=70, Tutoring_Sessions=2, Physical_Activity=3
    )
    print(f"Predicted Exam_Score for this student: {sample_score:.2f}")