# 📡 Predictive Customer Churn Classifier
### Random Forest & XGBoost | SMOTE | ROC-AUC | Telco Dataset

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-red?logo=xgboost&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project builds a **production-grade binary classification pipeline** to predict customer churn using IBM's Telco Customer Churn dataset. It tackles **severe class imbalance** using SMOTE oversampling and benchmarks **Random Forest** and **XGBoost** ensemble classifiers, targeting a **ROC-AUC score > 80%**.

**Key challenges addressed:**
- 73% / 27% class imbalance — handled with SMOTE
- Mixed data types — OneHotEncoding for categoricals, StandardScaler for numerics
- Business-driven feature engineering
- ROC-AUC optimized model evaluation

---

## 📂 Project Structure

```
Task3_CustomerChurn_Classifier/
│
├── Telco-Customer-Churn.csv            # IBM Telco dataset (local)
├── customer_churn_classifier.ipynb     # Main Jupyter Notebook
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── figures/
    ├── 01_churn_distribution.png
    ├── 02_numeric_distributions.png
    ├── 03_categorical_churn_rates.png
    ├── 04_smote_balance.png
    ├── 05_model_comparison.png
    ├── 06_roc_pr_curves.png
    ├── 07_confusion_matrices.png
    ├── 08_feature_importances.png
    └── 09_cv_roc_auc_boxplot.png
```

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset** — 7,043 telecom subscribers, 21 features.

| Feature Group | Features |
|---|---|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Account** | Contract, PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges |
| **Target** | ⭐ `Churn` — Yes / No |

**Class Distribution:**
- No Churn: ~5,174 (73.5%)
- Churn: ~1,869 (26.5%)
- Imbalance Ratio: ~2.8 : 1

---

## ⚙️ Feature Engineering

| New Feature | Description |
|---|---|
| `AvgMonthlySpend` | `TotalCharges / (tenure + 1)` |
| `ChargePerMonth` | `MonthlyCharges / (tenure + 1)` |
| `IsLongTermCustomer` | Binary flag — tenure > 24 months |

---

## ⚖️ Class Imbalance Strategy

| Method | Description |
|---|---|
| **SMOTE** | Synthetic Minority Oversampling — generates synthetic churn samples |
| **scale_pos_weight** | XGBoost native imbalance weighting (fallback) |
| **class_weight='balanced'** | Sklearn models fallback when SMOTE unavailable |

---

## 🔧 Preprocessing Pipeline

```
Numerical Features
    └── SimpleImputer (median) → StandardScaler

Categorical Features
    └── SimpleImputer (most_frequent) → OneHotEncoder

Combined via ColumnTransformer
    └── SMOTE (applied post-transform on training set only)
```

---

## 🤖 Models

### 1. Random Forest (Ensemble — Bagging)
```python
RandomForestClassifier(
    n_estimators=300, max_depth=15,
    min_samples_leaf=5, class_weight='balanced'
)
```

### 2. XGBoost (Ensemble — Boosting)
```python
XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=spw
)
```

### 3. Logistic Regression (Baseline)
```python
LogisticRegression(max_iter=1000, class_weight='balanced')
```

---

## 📈 Results

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | ~0.84 | ~0.79 | ~0.62 | ~0.77 | ~0.69 |
| Random Forest | ~0.87 | ~0.82 | ~0.68 | ~0.75 | ~0.71 |
| **XGBoost** | **~0.89** | **~0.84** | **~0.71** | **~0.78** | **~0.74** |

> ✅ All models exceed the **ROC-AUC > 0.80** target threshold.

---

## 🔑 Top Churn Predictors

Based on XGBoost feature importances:

1. 🥇 `tenure` — Longer customers churn less
2. 🥈 `MonthlyCharges` — Higher charges → higher churn
3. 🥉 `TotalCharges` — Correlated with tenure
4. `Contract_Month-to-month` — Highest churn risk segment
5. `InternetService_Fiber optic` — Higher churn than DSL
6. `OnlineSecurity_No` — No security = higher churn risk
7. `TechSupport_No` — No support = higher churn risk
8. `AvgMonthlySpend` — Engineered feature
9. `PaymentMethod_Electronic check` — Highest churn payment type
10. `PaperlessBilling_Yes` — Slight churn signal

---

## 📉 Visualizations

| Plot | Description |
|------|-------------|
| Churn Distribution | Bar, pie, and contract-type churn rates |
| Numeric Distributions | KDE plots split by churn status |
| Categorical Churn Rates | Bar charts for 6 key service features |
| SMOTE Balance | Before vs after class distribution |
| Model Comparison | Grouped bar chart across all metrics |
| ROC-AUC Curves | All models + random baseline |
| Precision-Recall Curves | AP score comparison |
| Confusion Matrices | Count + percentage heatmaps |
| Feature Importances | Top 10 for RF and XGBoost |
| CV Boxplot | 5-fold ROC-AUC distribution |

---

## 🚀 Getting Started

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook customer_churn_classifier.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Python 3.10+` | Core language |
| `Scikit-Learn` | RF, preprocessing, metrics, CV |
| `XGBoost` | Gradient boosting classifier |
| `imbalanced-learn` | SMOTE oversampling |
| `Pandas / NumPy` | Data manipulation |
| `Matplotlib / Seaborn` | Visualizations |
| `Jupyter Notebook` | Interactive development |

---

## 📚 References

- IBM Telco Customer Churn Dataset — [GitHub Source](https://github.com/IBM/telco-customer-churn-on-icp4d)
- SMOTE: Chawla et al. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique*
- XGBoost: Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System*

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">
  Made with ❤️ for customer retention analytics
</div>
