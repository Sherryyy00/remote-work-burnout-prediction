# Remote Work Burnout Prediction — Machine Learning Pipeline

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
## Code Screenshots
<img width="937" height="761" alt="1" src="https://github.com/user-attachments/assets/ccd68dee-3e15-4265-af2d-a7800c2ba22a" />
<img width="1331" height="730" alt="2" src="https://github.com/user-attachments/assets/8fa6b937-827c-402d-9d91-d9388cce9337" />
<img width="1173" height="767" alt="3" src="https://github.com/user-attachments/assets/df17d9ad-c236-4da4-b084-2c28e630e293" />
<img width="765" height="410" alt="4" src="https://github.com/user-attachments/assets/b59bd10f-9776-4fbe-8102-2b7c1c282674" />
<img width="1407" height="760" alt="4 1" src="https://github.com/user-attachments/assets/0c957e7e-bf50-4149-8f44-13dba8098a0f" />
<img width="546" height="596" alt="5" src="https://github.com/user-attachments/assets/73156fa0-0d4f-4b9f-b255-b918e265d514" />
<img width="1331" height="766" alt="6" src="https://github.com/user-attachments/assets/3ade9674-1ea7-4194-87e3-f8860f7a72f5" />
<img width="957" height="717" alt="7" src="https://github.com/user-attachments/assets/b637ee2c-13d0-4188-a6c5-5b1bcd3e07eb" />
<img width="1485" height="692" alt="8" src="https://github.com/user-attachments/assets/352c9de7-afa0-4644-a4bc-3cbcc85ae177" />
<img width="668" height="670" alt="9" src="https://github.com/user-attachments/assets/56a999e5-73c1-4298-8a4d-81c1a044c328" />
<img width="682" height="580" alt="10" src="https://github.com/user-attachments/assets/ceec4ad4-443b-4b9d-a9df-1a7f57d2181f" />
<img width="1003" height="392" alt="11" src="https://github.com/user-attachments/assets/c7198c3f-b469-4f94-8059-92b99ca01086" />
<img width="773" height="593" alt="12" src="https://github.com/user-attachments/assets/4ca46216-45f8-4c14-9489-62f4b35be327" />
<img width="842" height="547" alt="13" src="https://github.com/user-attachments/assets/620bdd33-9db3-4657-8168-6e4bac52f10a" />
<img width="766" height="401" alt="14" src="https://github.com/user-attachments/assets/5f6eff7c-7d92-40ae-866f-47b568b53928" />
<img width="662" height="527" alt="15" src="https://github.com/user-attachments/assets/4d04fdd1-a948-4783-90bb-5c1e19a88c8e" />
<img width="628" height="730" alt="16" src="https://github.com/user-attachments/assets/d39c6720-e33a-4152-af69-f3ec70091a57" />
<img width="1312" height="733" alt="17" src="https://github.com/user-attachments/assets/bc452475-5a1a-44bb-8dcf-a9268e0a1439" />
<img width="672" height="743" alt="18" src="https://github.com/user-attachments/assets/6692323b-4d1b-4275-9b72-1f30a05e217b" />
<img width="766" height="402" alt="19" src="https://github.com/user-attachments/assets/a1d5f94f-15ce-4adc-8f4d-861ae87ed60b" />
<img width="715" height="645" alt="20" src="https://github.com/user-attachments/assets/f7f186f2-c8ab-4f00-9adc-83200b9d2fdd" />
<img width="712" height="511" alt="21" src="https://github.com/user-attachments/assets/233f1cd5-c862-4d75-847c-dbfe9415b464" />
<img width="1032" height="658" alt="22" src="https://github.com/user-attachments/assets/74fd6cb4-3225-4423-a3f8-944e4ffb3953" />
<img width="572" height="503" alt="23" src="https://github.com/user-attachments/assets/f83ffe8b-c675-4917-ba9a-5dd85399f49d" />
<img width="857" height="625" alt="24" src="https://github.com/user-attachments/assets/fd516666-75c3-47af-97c7-38db2077e291" />
<img width="691" height="695" alt="25" src="https://github.com/user-attachments/assets/ebe8460a-b771-40f5-8995-85daa4ca6d82" />
<img width="726" height="766" alt="26" src="https://github.com/user-attachments/assets/74dc87b4-ad3a-4fb1-855b-713ffdee4101" />

---
## Author

**Muhammad Sheryar Adil**

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
