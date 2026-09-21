# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using customer data and supervised classification algorithms.

## 🎯 Objective

The goal of this project is to predict whether a customer is likely to churn, allowing businesses to identify at-risk customers and potentially improve customer retention.

## 🔄 Machine Learning Pipeline

The project follows a complete machine learning workflow:

1. Data loading and exploration
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature preparation
5. Train/test split
6. Model training
7. Cross-validation
8. Model comparison using ROC-AUC
9. Best model selection
10. Model pipeline serialization

## 🤖 Models Evaluated

| Model | Mean ROC-AUC | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.7876 | 0.0244 |
| Random Forest | 0.8478 | 0.0148 |
| Gradient Boosting | **0.8628** | **0.0102** |
| AdaBoost | 0.8462 | 0.0133 |
| SVC | 0.8346 | 0.0104 |

## 🏆 Best Model

**Gradient Boosting** achieved the highest mean ROC-AUC:

**ROC-AUC: 0.8628 ± 0.0102**

The relatively low standard deviation also indicates consistent performance across the cross-validation folds.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Joblib

## 📁 Repository Structure

```text
customer-churn-prediction/
│
├── Analysis.ipynb
├── customer_data.csv
├── best_model_pipeline.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/OSama2626/customer-churn-prediction.git
cd customer-churn-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```bash
Analysis.ipynb
```

to explore the complete analysis and machine learning workflow.

## 👤 Author

**Oussama Abouhafs**

Software Engineer | Data & Machine Learning
