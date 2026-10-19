# Student Wellbeing Score

[![Live demo](https://img.shields.io/badge/demo-live-3b6fd8)](https://abdullahrasheed452.github.io/MentalHealthScorePredictor/)
![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-f7931e)
![No backend](https://img.shields.io/badge/backend-none-555)
[![License: MIT](https://img.shields.io/badge/license-MIT-2e8b57)](LICENSE)

A privacy-first web app that estimates a student's wellbeing on a 0–100 scale from eight questions
about their studies and habits, and shows which answers are helping or hurting the score. The model is
trained on 27,837 student survey responses and validated on a separate university survey.

**[Try the live demo →](https://abdullahrasheed452.github.io/MentalHealthScorePredictor/)**

> [!IMPORTANT]
> This is an educational project, not a medical tool. The score is a statistical estimate and can be
> wrong for any individual. If you are struggling, please talk to someone you trust, your university
> counselling service, or a doctor.

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Results](#results)
- [Limitations](#limitations)
- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Data sources](#data-sources)
- [License](#license)

## Features

- **Private by design.** The model runs entirely in the browser. There is no server, and nothing the
  user enters leaves their device.
- **Explainable.** Each answer is shown as a bar with the number of points it adds or removes,
  alongside practical tips for the factors pulling the score down most.
- **Validated beyond its training data.** Tested unchanged on a second, independent student survey.
- **Responsive and accessible.** Works on phones and desktops, in light and dark mode, with keyboard
  support and colour choices checked for colour-blind readers.

## How it works

```
Kaggle survey ──▶ src/train.py ──▶ docs/model.json ──▶ browser (docs/app.js) ──▶ score + factors
                  logistic regression    weights           score = 100 × (1 − risk)
```

1. `src/train.py` trains a logistic regression to predict the probability that a student reports
   depression, and exports its weights to `docs/model.json`.
2. The static site in `docs/` loads those weights and computes the score in JavaScript.
3. Each factor's bar shows how much that answer, on its own, would change an average student's score.

| Input | Form | Notes |
|---|---|---|
| Academic pressure | 1–5 slider | Strongest single factor |
| Financial stress | 1–5 slider | Second strongest |
| Study satisfaction | 1–5 slider | |
| Study / work hours | 0–12 hours | |
| Sleep | 4 options | |
| Diet | 3 options | |
| CGPA | out of 4.0 | Converted to the survey's 10-point scale |
| Family history of mental illness | yes / no | |

Logistic regression was chosen because it matched more complex models on accuracy (see below), runs in
a few lines of JavaScript, and makes each input's effect directly explainable.

## Results

### Model comparison

5-fold cross-validation on 27,837 students (`src/compare_models.py`):

| Model | AUC | Accuracy |
|---|---|---|
| **Logistic regression (used by the site)** | **0.860** | **78.8%** |
| Logistic regression + pairwise interactions | 0.859 | 78.8% |
| Gradient boosting (tuned) | 0.858 | 78.7% |
| Random forest | 0.855 | 78.5% |

A learning curve shows the model reaches AUC 0.859 with only 5,000 training rows, so more data of the
same kind would not improve it. Better accuracy would need different information about each student.

### External validation

The training survey comes from India, so the model was tested unchanged on the
[PHQ-9 Student Depression Dataset](https://data.mendeley.com/datasets/kkzjk253cy/6) (682 university
students), which uses different questions and measures depression with the clinical PHQ-9
questionnaire (score ≥ 10 = moderate depression or worse). The two surveys share two inputs:

| Using academic + financial pressure only | AUC |
|---|---|
| Training survey (India) | 0.827 |
| Independent survey (PHQ-9) | **0.830** |

The model ranks students by risk as well on the independent survey as on its own.
`src/validate_external.py` checks data quality before testing (Cronbach's α = 0.88); a third dataset
that failed similar consistency checks was excluded.

### Feature decisions

| Left out | Reason |
|---|---|
| Suicidal thoughts | Raises AUC to ≈0.92, but too sensitive for a public form, and partly a symptom of depression itself |
| Age | Its effect did not replicate on the independent survey; removing it improved external AUC (0.818 → 0.830) |
| Gender, degree, city | No accuracy gain, and not things a student can change |

## Limitations

- **Population.** Trained on students in India. One independent survey is encouraging, not proof of
  broad generalisation.
- **Self-reported data.** Both habits and the depression label come from questionnaires, not clinical
  assessment.
- **Correlation, not causation.** The model cannot say that changing a habit will change how someone
  feels.
- **High base rate.** 58.5% of the training survey reported depression, well above typical rates, so
  scores tend to run low. The site hides the score until the user answers and keeps it within 1–99.

## Getting started

**Requirements:** Python 3.10+

```bash
git clone https://github.com/AbdullahRasheed452/MentalHealthScorePredictor.git
cd MentalHealthScorePredictor
pip install -r requirements.txt
```

Download the [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
CSV into `data/raw/`, then:

```bash
python src/train.py                 # train and export docs/model.json
python src/compare_models.py        # optional: model comparison and learning curve
python src/validate_external.py     # optional: external validation (PHQ-9 CSV in data/external/)

python -m http.server 8000 --directory docs   # open http://localhost:8000
```

### Deployment

GitHub Pages serves `docs/` from the `main` branch (Settings → Pages → *Deploy from a branch* →
`main` / `/docs`). After retraining, commit the new `docs/model.json` and push.

## Project structure

```
├── data/
│   ├── raw/                  training CSV (not committed)
│   └── external/             validation CSV (not committed)
├── docs/                     the website, served by GitHub Pages
│   ├── index.html
│   ├── app.js                scoring and UI logic
│   ├── style.css
│   └── model.json            exported model weights
├── models/                   evaluation results (JSON)
├── src/
│   ├── train.py              training and export
│   ├── compare_models.py     model comparison and learning curve
│   └── validate_external.py  external validation
├── LICENSE
└── requirements.txt
```

## Data sources

- [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) by hopesb, Kaggle.
- [PHQ-9 Student Depression Dataset](https://data.mendeley.com/datasets/kkzjk253cy/6), Mendeley Data, licensed CC BY 4.0.

## License

The code is released under the [MIT License](LICENSE). The datasets are not included in this
repository and remain under their own terms; see [Data sources](#data-sources).
