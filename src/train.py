"""Train the student wellbeing model and export it for the static web app.

Usage:
    python src/train.py                 # uses the first CSV found in data/raw/
    python src/train.py path/to/file.csv

Outputs:
    docs/model.json     weights the browser uses to compute the score
    models/metrics.json evaluation results for the README
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TARGET = "Depression"

NUMERIC = ["Academic Pressure", "CGPA", "Study Satisfaction",
           "Work/Study Hours", "Financial Stress"]
SLEEP_LEVELS = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
DIET_LEVELS = ["Healthy", "Moderate", "Unhealthy"]
FAMILY = "Family History of Mental Illness"

# Deliberately left out: suicidal-thoughts question (too sensitive to ask in a
# public web form), and City / Degree / Gender (not things a student can change,
# and they add no accuracy). Age is also left out: it can't be changed, and its
# effect did not hold up on students outside India (see src/validate_external.py).


def find_csv() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    csvs = sorted((ROOT / "data" / "raw").glob("*.csv"))
    if not csvs:
        sys.exit("No CSV in data/raw/. Download the Kaggle dataset first (see README).")
    return csvs[0]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    needed = NUMERIC + ["Sleep Duration", "Dietary Habits", FAMILY, TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing expected columns: {missing}\nFound: {list(df.columns)}")

    if "Profession" in df.columns:
        df = df[df["Profession"].str.strip() == "Student"]

    for col in ["Sleep Duration", "Dietary Habits", FAMILY]:
        df[col] = df[col].astype(str).str.strip().str.strip("'\"")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Sleep Duration"].isin(SLEEP_LEVELS) & df["Dietary Habits"].isin(DIET_LEVELS)]
    df = df.dropna(subset=NUMERIC + [TARGET])
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into the numeric features the model (and the browser) use."""
    X = df[NUMERIC].astype(float).copy()
    # Reference levels (dropped): "7-8 hours" sleep and "Healthy" diet.
    for level in SLEEP_LEVELS:
        if level != "7-8 hours":
            X[f"Sleep: {level}"] = (df["Sleep Duration"] == level).astype(float)
    for level in DIET_LEVELS:
        if level != "Healthy":
            X[f"Diet: {level}"] = (df["Dietary Habits"] == level).astype(float)
    X["Family history"] = (df[FAMILY] == "Yes").astype(float)
    return X


def main():
    path = find_csv()
    df = load(path)
    X = encode(df)
    y = df[TARGET].astype(int).to_numpy()
    print(f"Loaded {len(df):,} student rows from {path.name} "
          f"({y.mean():.1%} labelled as depression)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    logreg = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    p_lr = logreg.predict_proba(scaler.transform(X_test))[:, 1]

    gbm = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    p_gb = gbm.predict_proba(X_test)[:, 1]

    metrics = {
        "rows": int(len(df)),
        "logistic_regression": {"auc": roc_auc_score(y_test, p_lr),
                                "accuracy": accuracy_score(y_test, p_lr >= 0.5)},
        "gradient_boosting": {"auc": roc_auc_score(y_test, p_gb),
                              "accuracy": accuracy_score(y_test, p_gb >= 0.5)},
    }
    for name in ("logistic_regression", "gradient_boosting"):
        m = metrics[name]
        print(f"  {name:20s} AUC {m['auc']:.3f}  accuracy {m['accuracy']:.3f}")

    # The web app ships logistic regression: it runs in a few lines of JS and
    # each feature's contribution to the score can be shown directly.
    model = {
        "intercept": float(logreg.intercept_[0]),
        "features": [
            {"name": col, "mean": float(mu), "std": float(sd), "coef": float(w)}
            for col, mu, sd, w in zip(X.columns, scaler.mean_, scaler.scale_, logreg.coef_[0])
        ],
        "trained_on_rows": int(len(df)),
        "test_auc": round(metrics["logistic_regression"]["auc"], 3),
    }

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "docs" / "model.json").write_text(json.dumps(model, indent=2))
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Wrote docs/model.json and models/metrics.json")


if __name__ == "__main__":
    np.random.seed(42)
    main()
