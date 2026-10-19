"""Test the site's model on students from a different survey (external validation).

Uses the PHQ-9 Student Depression Dataset (Mendeley Data, CC BY 4.0):
https://data.mendeley.com/datasets/kkzjk253cy/6

Usage:
    python src/validate_external.py                 # uses the CSV in data/external/
    python src/validate_external.py path/to/file.csv

Output:
    models/external_validation.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train

ROOT = Path(__file__).resolve().parent.parent
PHQ_ANSWERS = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
# "Good / Average / Bad / Worst" mapped onto the site's 1-5 sliders.
PRESSURE = {"Good": 2, "Average": 3, "Bad": 4, "Worst": 5}
# Only these site inputs have a counterpart in the PHQ-9 dataset. The others are
# set to the training average, so they contribute nothing to the prediction.
SHARED = ["Academic Pressure", "Financial Stress"]


def find_csv() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for path in sorted((ROOT / "data" / "external").glob("*.csv")):
        if "PHQ_Total" in pd.read_csv(path, nrows=0).columns:
            return path
    sys.exit("No PHQ-9 CSV in data/external/. Download it first (see README).")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def check_consistency(df: pd.DataFrame) -> dict:
    """Real PHQ-9 data has questions that agree with each other. Fabricated or
    scrambled data doesn't, so check before trusting any result."""
    answers = df[df.columns[2:11]].apply(lambda s: s.str.strip().map(PHQ_ANSWERS))
    if answers.isna().any().any():
        sys.exit("Unexpected answers in the PHQ-9 columns.")
    corr = answers.corr(method="spearman").to_numpy()
    k = answers.shape[1]
    alpha = k / (k - 1) * (1 - answers.var().sum() / answers.sum(axis=1).var())
    return {
        "total_matches_answers": float((answers.sum(axis=1) == df["PHQ_Total"]).mean()),
        "mean_question_correlation": float(corr[~np.eye(k, dtype=bool)].mean()),
        "cronbach_alpha": float(alpha),
    }


def site_risk(X: pd.DataFrame) -> np.ndarray:
    """Exactly what docs/app.js computes, with unknown inputs left at the average."""
    model = json.loads((ROOT / "docs" / "model.json").read_text())
    logit = model["intercept"]
    for f in model["features"]:
        if f["name"] in X:
            logit = logit + f["coef"] * (X[f["name"]] - f["mean"]) / f["std"]
    return 1 / (1 + np.exp(-logit))


def india_reference() -> float | None:
    """AUC on Indian students when the model only gets the same shared inputs."""
    csvs = sorted((ROOT / "data" / "raw").glob("*.csv"))
    if not csvs:
        return None
    df = train.load(csvs[0])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    p = cross_val_predict(lr, df[SHARED], df[train.TARGET].astype(int), cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(df[train.TARGET].astype(int), p))


def main():
    path = find_csv()
    df = load(path)
    checks = check_consistency(df)
    print(f"Loaded {len(df):,} students from {path.name}")
    print(f"  PHQ total matches answers: {checks['total_matches_answers']:.0%}   "
          f"question correlation {checks['mean_question_correlation']:+.2f}   "
          f"Cronbach's alpha {checks['cronbach_alpha']:.2f}")
    if checks["cronbach_alpha"] < 0.7:
        sys.exit("The PHQ-9 answers don't agree with each other. This data can't be trusted.")

    # PHQ-9 total of 10 or more is the standard cut-off for moderate depression.
    y = (df["PHQ_Total"] >= 10).astype(int)
    X = pd.DataFrame({
        "Academic Pressure": df["Study Pressure"].str.strip().map(PRESSURE),
        "Financial Stress": df["Financial Pressure"].str.strip().map(PRESSURE),
    })
    risk = site_risk(X)
    results = {
        "rows": int(len(df)),
        "depressed_share": float(y.mean()),
        "consistency": checks,
        "site_model_auc": float(roc_auc_score(y, risk)),
        "site_model_spearman_vs_phq_total": float(spearmanr(risk, df["PHQ_Total"])[0]),
        "india_auc_same_inputs": india_reference(),
    }
    print(f"  {y.mean():.1%} have PHQ-9 >= 10 (moderate depression or worse)")
    print(f"  Site model on these students: AUC {results['site_model_auc']:.3f}")
    if results["india_auc_same_inputs"] is not None:
        print(f"  Same inputs on Indian students: AUC {results['india_auc_same_inputs']:.3f}")

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "external_validation.json").write_text(json.dumps(results, indent=2))
    print("Wrote models/external_validation.json")


if __name__ == "__main__":
    main()
