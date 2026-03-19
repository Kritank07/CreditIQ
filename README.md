# CreditIQ — Loan Approval Prediction System
An end-to-end machine learning pipeline that predicts loan approval outcomes based on applicant financial data, credit history, and employment details — achieving **87% accuracy**.

# Table of Contents
Overview
Dataset
Project Workflow
Models & Results
Feature Engineering
Tech Stack
How to Run
Project Structure
Key Insights

# Overview
CreditIQ is a supervised machine learning project built to automate and improve loan approval decisions in the banking/fintech sector. By analyzing applicant data such as income, credit score, DTI ratio, employment status, and more, the model predicts whether a loan should be approved or rejected.
This project covers the complete ML workflow — from raw data cleaning to model comparison and feature engineering.

# Dataset
Property	Details
Records	1,000 applicants
Features (after encoding)	28
Target Variable	`Loan_Approved` (Yes / No)
Missing Values	~50 per column (handled via imputation)
Key Features:
`Applicant_Income`, `Coapplicant_Income`
`Credit_Score`, `DTI_Ratio`, `Savings`
`Employment_Status`, `Marital_Status`, `Gender`
`Loan_Amount`, `Loan_Purpose`, `Collateral_Value`
`Age`, `Dependents`, `Education_Level`, `Property_Area`

# Project Workflow
Raw Data
   │
   ▼
Data Cleaning & Imputation
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Encoding (Label + One-Hot)
   │
   ▼
Correlation Heatmap
   │
   ▼
Train-Test Split (80/20)
   │
   ▼
Feature Scaling (StandardScaler)
   │
   ▼
Model Training & Evaluation
   │
   ▼
Feature Engineering
   │
   ▼
Re-training & Final Results

# Models & Results
Before Feature Engineering
Model	Accuracy	Precision	Recall
Logistic Regression	86.5%	78.3%	77.0%
Naive Bayes	86.5%	80.4%	73.8%
KNN (k=9)	76.0%	65.9%	44.3%
After Feature Engineering
Model	Accuracy	Precision	Recall
Logistic Regression	87.0%	77.8%	80.3%
Naive Bayes	86.5%	80.4%	73.8%
KNN (k=9)	77.0%	68.3%	45.9%
**Best Overall Accuracy:** Logistic Regression (87% after feature engineering)  
**Best Precision:** Naive Bayes (80.4%) — preferred when minimizing false approvals

# Feature Engineering
Three new features were created to boost model performance:
Feature	Transformation
`Credit_Score_sq`	Credit Score² (polynomial)
`DTI_Ratio_sq`	DTI Ratio² (polynomial)
`Applicant_Income_log`	log1p(Applicant Income) — reduces skewness

# Tech Stack
Language: Python 3.8+
Data Manipulation: Pandas, NumPy
Visualization: Seaborn, Matplotlib
Machine Learning: Scikit-learn
`LogisticRegression`
`GaussianNB`
`KNeighborsClassifier`
`StandardScaler`, `SimpleImputer`
`LabelEncoder`, `OneHotEncoder`

# How to Run
Clone the repository
```bash
   git clone https://github.com/your-username/CreditIQ.git
   cd CreditIQ
   ```
Install dependencies
```bash
   pip install pandas numpy seaborn matplotlib scikit-learn jupyter
   ```
Add the dataset
Place `loan_approval_data.csv` in the root directory.
Run the notebook
```bash
   jupyter notebook credit_wise.ipynb
   ```
---
# Project Structure
```
CreditIQ/
│
├── credit_wise.ipynb        # Main Jupyter Notebook
├── loan_approval_data.csv   # Dataset (add manually)
└── README.md                # Project documentation
```
---
# Key Insights
Credit Score and DTI Ratio are among the strongest predictors of loan approval.
Naive Bayes consistently delivered the best precision — ideal for conservative lending decisions where false approvals are costly.
Feature engineering (polynomial + log transforms) improved Logistic Regression accuracy from 86.5% → 87%.
The dataset was fairly imbalanced (~65% rejected, ~35% approved), which was considered during model evaluation.
