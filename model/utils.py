import pandas as pd

# ======================================================
# 1. Basic Data Validation (Optional but Professional)
# ======================================================
def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures input dataframe has correct columns and types.
    Used before prediction in Flask.
    """
    required_columns = [
        "male", "age", "education", "currentSmoker", "cigsPerDay",
        "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


# ======================================================
# 2. Light Feature Preparation (NO ENGINEERING)
# ======================================================
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal preparation:
    - Copy dataframe
    - Ensure numeric types
    - No feature engineering
    """
    df = df.copy()

    numeric_cols = [
        "age", "education", "cigsPerDay", "totChol",
        "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    binary_cols = [
        "male", "currentSmoker", "BPMeds",
        "prevalentStroke", "prevalentHyp", "diabetes"
    ]

    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
