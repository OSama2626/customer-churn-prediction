# Customer Churn Prediction

A Streamlit application for exploring customer churn risk using saved, version 2 machine learning pipelines. It supports individual customer scoring, local and global SHAP explanations, model evaluation details, and batch CSV scoring.

## What the application does

- Scores a customer from 10 raw fields using a saved pipeline that performs its own feature engineering and preprocessing.
- Shows churn probability, the active model, decision threshold, classification, and a simple risk level.
- Explains local predictions and global model behavior with SHAP values, translated into business labels.
- Validates uploaded customer CSVs, displays batch predictions, and lets users download the results.
- Presents baseline evaluation metrics from the saved baseline metadata.

The reference model is the saved Gradient Boosting pipeline. The saved XGBoost pipeline is available as a detection-focused alternative. The XGBoost metadata contains its threshold and model settings, but does not contain evaluation metrics; the application therefore does not report XGBoost performance figures.

## Install and run locally

Use Python 3.12 or a compatible version supported by the pinned packages. From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

Run the full test suite:

```bash
python -m pytest
```

## Models, probability, and threshold

The application loads the saved pipelines directly; it does not train or refit them. A probability is the model's estimated score for churn. It is not presented as a calibrated probability of a real-world outcome.

The classification rule is `likely churn` when the churn probability is greater than or equal to the selected threshold. Each model's default threshold comes from its own metadata:

| Model | Default threshold | Source |
|---|---:|---|
| Gradient Boosting (reference) | 0.50 | `baseline_v2_metadata.json` |
| XGBoost (detection alternative) | 0.30 | `xgboost_v2_metadata.json` |

On the prediction page, XGBoost can also be scored with the 0.50 alternate threshold. The active model and threshold are shown before scoring and applied consistently to individual and batch predictions. A lower threshold marks more customers as likely to churn; it can increase detection while also creating more false alarms. Threshold changes affect the class decision, not the underlying model probability or ROC-AUC.

## Evaluation and methodology

The reference pipeline accepts these raw fields: credit score, country, gender, age, tenure, balance, products, credit card, active-member status, and estimated salary. It derives balance per product, a salary-to-balance ratio, age and tenure groups, and a high-balance indicator inside the fitted pipeline. Imputation, scaling, and categorical encoding are also part of the pipeline.

The saved baseline metadata records a stratified 80/20 split and five-fold stratified cross-validation. Its results are:

| Metric | Recorded result |
|---|---:|
| CV ROC-AUC | 0.8630 ± 0.0103 |
| Held-out test ROC-AUC | 0.8716 |
| Precision | 0.7756 |
| Recall | 0.4840 |
| F1 | 0.5961 |
| Accuracy | 0.8665 |

The precision, recall, F1, and accuracy values correspond to the baseline metadata threshold of 0.50. ROC-AUC describes ranking performance over thresholds. These are recorded project results, not a promise of future performance. The held-out test set was used to compare project variants, so it is not independent external validation. The input probabilities have not been calibrated. SHAP describes model contributions and does not establish causation.

## Project structure

```text
app.py                         Streamlit overview page
pages/                         Prediction, performance, insights, methodology
src/config.py                  Artifact paths and metadata access
src/predictor.py               Cached model loading and single inference
src/batch.py                   CSV schema validation and batch inference
src/validation.py              Raw customer validation
src/explainability.py          SHAP explanations and business labels
src/forms.py                   Reusable customer form
src/ui.py                      Shared styling and page components
assets/app.css                 Application styles
tests/                         Model and Streamlit tests
customer_data.csv              Source customer dataset
baseline_v2_pipeline.pkl       Saved Gradient Boosting pipeline
baseline_v2_metadata.json      Baseline schema, threshold, and metrics
xgboost_v2_pipeline.pkl        Saved XGBoost pipeline
xgboost_v2_metadata.json       XGBoost schema, threshold, and settings
shap_global_importance.csv     Saved global SHAP summary
export_v2_artifact.py          Baseline artifact export script
export_optional_artifacts.py   Optional XGBoost and SHAP export script
```

The export scripts are included for reproducibility; they are not needed to run the application.

`Analysis.ipynb` contains the original analysis. The Streamlit application loads the checked-in artifacts and does not need to run the notebook.

## Known limitations

- The evaluation is based on a static dataset and may not represent future customers.
- The held-out evaluation is not independent external validation.
- Probabilities are not calibrated for operational decisions.
- SHAP values explain model behavior over transformed features, not causal effects.
- The XGBoost artifact currently has no evaluation metrics in its metadata, so it is offered for scoring without comparative performance claims.
