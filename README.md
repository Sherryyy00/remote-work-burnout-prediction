# 🔥 Remote Work Burnout Prediction — Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete end-to-end machine learning pipeline to **predict employee burnout risk** in remote working environments. This project covers exploratory data analysis, class imbalance handling, multi-model classification, regression, threshold tuning, cross-validation, and feature importance analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Models Used](#models-used)
- [Key Results](#key-results)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Technologies](#technologies)
- [Author](#author)

---

## Overview

Remote work has increased significantly in recent years, and with it the risk of employee burnout. This project builds a machine learning system that:

- **Classifies** employees into burnout risk categories (Low or High)
- **Predicts** a continuous burnout score using regression
- **Identifies** the behavioural features that contribute most to burnout risk

A key challenge addressed in this project is **severe class imbalance** — High-risk cases form a small minority of the dataset. The pipeline includes SMOTE oversampling, class weighting, and decision threshold tuning to ensure the models are genuinely useful rather than superficially accurate.

---

## Dataset

**File:** `work_from_home_burnout_dataset.csv`

| Feature | Type | Description |
|---|---|---|
| `work_hours` | Continuous | Daily hours worked |
| `screen_time_hours` | Continuous | Daily screen time in hours |
| `meetings_count` | Integer | Number of meetings per day |
| `breaks_taken` | Integer | Number of breaks taken per day |
| `after_hours_work` | Binary | Whether the employee worked after hours (0 or 1) |
| `sleep_hours` | Continuous | Hours of sleep per night |
| `task_completion_rate` | Continuous | Percentage of daily tasks completed |
| `day_type` | Categorical | Weekday or Weekend |
| `burnout_score` | Continuous | **Target (regression):** numerical burnout score |
| `burnout_risk` | Categorical | **Target (classification):** Low or High |

> **1,800 records** across 10 features.

---

## Project Structure

```
remote-work-burnout-prediction/
│
├── Remote_Work_Burnout_Improved.ipynb   # ✅ Full improved pipeline (recommended)
├── Remote_Work_Burnout_Prediction.ipynb # Original notebook
│
├── work_from_home_burnout_dataset.csv   # Dataset (add your own copy here)
│
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Files excluded from version control
├── LICENSE                              # MIT License
└── README.md                            # This file
```

---

## Pipeline Stages

```
Raw Data
   ↓
Exploratory Data Analysis (distributions, heatmaps, box plots)
   ↓
Preprocessing (LabelEncoder, StandardScaler, null checks)
   ↓
SMOTE — Synthetic Minority Oversampling
   ↓
Model Training (RF Classifier, Logistic Regression, Gradient Boosting)
   ↓
Threshold Tuning (F1-optimal decision boundary for RF)
   ↓
Evaluation (Confusion Matrix, ROC-AUC, Classification Report)
   ↓
Cross-Validation (5-fold Stratified)
   ↓
Feature Importance Analysis
   ↓
Regression (RF Regressor → burnout score)
   ↓
Sample Prediction (new employee profile)
```

---

## Models Used

### Classification (Burnout Risk: Low / High)

| Model | Key Settings |
|---|---|
| Random Forest Classifier | 200 estimators, class_weight=balanced, threshold tuned |
| Logistic Regression | max_iter=1000, class_weight=balanced |
| Gradient Boosting Classifier | 200 estimators, lr=0.05, max_depth=4 |

### Regression (Burnout Score)

| Model | Key Settings |
|---|---|
| Random Forest Regressor | 200 estimators |

---

## Key Results

> Note: Exact values depend on random state and SMOTE sampling. Values below are representative.

| Model | Accuracy | F1 (High class) | ROC-AUC |
|---|---|---|---|
| Random Forest (tuned) | ~0.92 | Improved | ~0.97 |
| Logistic Regression | ~0.89 | Moderate | ~0.94 |
| Gradient Boosting | ~0.94 | Strong | ~0.98 |

**Regression (Burnout Score)**
- R² score demonstrates strong predictive fit
- Visualised with predicted vs actual scatter plot and residuals plot

**Critical fix applied:** Without threshold tuning, the Random Forest achieved 97.5% accuracy but detected **zero** High-risk cases. After tuning, High-risk recall improved substantially — accuracy alone is not a valid metric for imbalanced classification.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/remote-work-burnout-prediction.git
cd remote-work-burnout-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place your `work_from_home_burnout_dataset.csv` file in the root of the repository.

### 5. Launch Jupyter

```bash
jupyter notebook
```

---

## Usage

Open `Remote_Work_Burnout_Improved.ipynb` and run all cells from top to bottom. The notebook is fully self-contained and will:

1. Load and explore the dataset
2. Preprocess features and handle class imbalance
3. Train and evaluate all models
4. Display visualisations for each stage
5. Predict burnout risk for a sample employee profile
6. Print a full project summary

To predict for a custom employee, edit the sample dictionary in **Section 15**:

```python
sample = pd.DataFrame({
    'day_type_enc':         [1],      # 1 = Weekday, 0 = Weekend
    'work_hours':           [9.5],
    'screen_time_hours':    [12.2],
    'meetings_count':       [5],
    'breaks_taken':         [1],
    'after_hours_work':     [1],
    'sleep_hours':          [5.6],
    'task_completion_rate': [60]
})
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `Remote_Work_Burnout_Improved.ipynb` | Full pipeline with all fixes and additions — **start here** |
| `Remote_Work_Burnout_Prediction.ipynb` | Original exploratory notebook |

---

## Technologies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Visualisation |
| `seaborn` | Statistical plots and heatmaps |
| `scikit-learn` | Preprocessing, models, and evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `jupyter` | Interactive notebook environment |

---

## Author

**Muhammad Sheryar Adil**
BSc Computer Science / Data Science — Teesside University
Research Internship — Digital Building, Teesside University (Jan 2026 – May 2026)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
