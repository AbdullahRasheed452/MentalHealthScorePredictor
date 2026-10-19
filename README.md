# Student Wellbeing Score

A web app that estimates a 0–100 wellbeing score for students from study and lifestyle habits:
academic pressure, study satisfaction, study hours, financial stress, sleep, diet, CGPA, age and
family history.

**Live demo:** https://abdullahrasheed452.github.io/MentalHealthScorePredictor/

> This is an educational project, not a medical tool. The score is a statistical estimate and can
> be wrong for any individual.

## How it works

1. `src/train.py` trains a logistic regression on the Kaggle
   [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
   to predict the probability that a student reports depression. It also trains a gradient-boosting
   model for comparison.
2. The logistic regression weights are exported to `docs/model.json`.
3. The static site in `docs/` loads those weights and computes
   **score = 100 × (1 − predicted probability)** in the browser. No server is needed, and nothing the
   user enters leaves their device.
4. Each input's contribution to the prediction is shown as a factor that's helping or hurting the score.

## Results

Trained on 27,837 student rows (58.5% labelled as depression), with a stratified 80/20 test split:

| Model | Test AUC | Accuracy |
|---|---|---|
| Logistic regression (used by the site) | 0.868 | 79.5% |
| Gradient boosting (for comparison) | 0.864 | 79.3% |

Logistic regression is used for the site because it runs in a few lines of JavaScript and its
per-feature contributions are directly explainable. The question about suicidal thoughts in the
original dataset is left out on purpose: it is too sensitive to ask in a public web form.

## Run it yourself

```bash
pip install -r requirements.txt

# 1. Download the dataset from Kaggle and put the CSV in data/raw/
# 2. Train and export the model
python src/train.py

# 3. Serve the site locally, then open http://localhost:8000
python -m http.server 8000 --directory docs
```

## Project layout

```
data/raw/        Kaggle CSV (not committed)
src/train.py     training + export
models/          evaluation metrics
docs/            the website (served by GitHub Pages)
```

## Deploying

GitHub Pages serves the `docs/` folder from the `main` branch
(Settings → Pages → Source: *Deploy from a branch* → `main` / `/docs`).
After retraining, commit the new `docs/model.json` and push. The site updates within a minute or two.
