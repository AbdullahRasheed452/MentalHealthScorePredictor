"""Check whether a more complex model, or more data, would beat the site's model.

Usage:
    python src/compare_models.py

Output:
    models/comparison.json
"""

import json
import warnings
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import train

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent.parent
CV = StratifiedKFold(5, shuffle=True, random_state=42)

MODELS = {
    "Logistic regression (site model)":
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Logistic regression + pairwise interactions":
        make_pipeline(StandardScaler(), PolynomialFeatures(2, interaction_only=True),
                      StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
    "Gradient boosting (tuned)":
        HistGradientBoostingClassifier(learning_rate=0.03, max_iter=600, max_leaf_nodes=15,
                                       l2_regularization=1.0, early_stopping=True, random_state=42),
    "Random forest":
        RandomForestClassifier(500, min_samples_leaf=10, n_jobs=-1, random_state=42),
}


def main():
    csvs = sorted((ROOT / "data" / "raw").glob("*.csv"))
    df = train.load(csvs[0])
    X, y = train.encode(df), df[train.TARGET].astype(int).to_numpy()
    print(f"5-fold cross-validation on {len(df):,} students\n")

    results = {"models": {}, "learning_curve": {}}
    for name, model in MODELS.items():
        p = cross_val_predict(model, X, y, cv=CV, method="predict_proba")[:, 1]
        auc, acc = roc_auc_score(y, p), accuracy_score(y, p >= 0.5)
        results["models"][name] = {"auc": auc, "accuracy": acc}
        print(f"  {name:45s} AUC {auc:.3f}  accuracy {acc:.3f}")

    # Would more rows of the same survey help? Train on growing subsets.
    print("\nTraining rows -> test AUC (site model)")
    sizes, _, test = learning_curve(MODELS["Logistic regression (site model)"], X, y, cv=CV,
                                    train_sizes=[500, 1000, 2000, 5000, 10000, 22000],
                                    scoring="roc_auc", shuffle=True, random_state=42)
    for n, auc in zip(sizes, test.mean(axis=1)):
        results["learning_curve"][int(n)] = float(auc)
        print(f"  {n:6d}  {auc:.3f}")

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "comparison.json").write_text(json.dumps(results, indent=2))
    print("\nWrote models/comparison.json")


if __name__ == "__main__":
    main()
