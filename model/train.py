import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from model.utils import prepare_features

df = pd.read_csv(r"D:\heart_disease\dataset\framingham.csv")
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]
X = prepare_features(X)

binary_cols = [
    "male", "currentSmoker", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes"
]

numeric_cols = [
    "age", "education", "cigsPerDay", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

binary_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("bin", binary_pipeline, binary_cols)
    ]
)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
joblib.dump(model, "model/framingham_logreg.joblib")
