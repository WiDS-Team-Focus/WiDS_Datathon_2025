# WiDS Datathon 2025 – Brain Connectome Modeling for ADHD and Sex Prediction

This repository contains a full data science workflow for the **WiDS Datathon 2025 Global Challenge**, which focuses on predicting two outcomes—**ADHD diagnosis** and **Sex (female/male)**—using functional brain imaging (fMRI) data and socio-demographic metadata from children and adolescents.

---

## 💡 Challenge Overview

**Goal**:  
Build a multi-output model to predict:
- `ADHD_Outcome`: 1 if the participant has ADHD, 0 otherwise  
- `Sex_F`: 1 if the participant is female, 0 if male

**Data Types**:
- Functional MRI (fMRI) connectome matrices (36 participants × 19900+ features)
- Quantitative metadata (emotions, parenting styles)
- Categorical metadata (handedness, parental education, etc.)
- Labels

**Evaluation Metric**:  
Weighted average of the F1 scores for both targets, with **double weight** assigned to female ADHD cases.

---

## 📁 Folder Structure

```
├── notebooks/
│   └── eda_and_modeling.ipynb       # Full notebook with EDA, XGBoost, TensorFlow model
├── data/
│   ├── TRAIN_OLD/
│   ├── TRAIN_NEW/
│   └── TEST/
├── README.md
└── submission/
    └── submission.csv
```

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked class imbalance (ADHD and Sex)
- Visualized correlation matrix for quantitative metadata
- Identified missing values across columns
- Inspected fMRI connectome feature distributions

---

## 🧠 Models Used

### 1. XGBoost Classifier (per target)

Each target (`ADHD_Outcome`, `Sex_F`) was modeled using:
- Feature selection (`SelectKBest`)
- Imputation and standardization
- Evaluation with F1 score

### 2. TensorFlow Neural Network (multi-output)

A simple feedforward neural network was built with:
- Shared dense layers
- Dual output heads (`sigmoid` for Sex, `relu` for ADHD)
- Optimized using binary crossentropy loss and Adam optimizer

---

## 🧪 Test Submission Preparation

- Merged test fMRI + metadata
- Applied same preprocessing as training
- Used trained models to predict `ADHD_Outcome` and `Sex_F`
- Formatted predictions to the required submission CSV:
  ```csv
  participant_id,ADHD_Outcome,Sex_F
  v1nMpCoLGU0V,0.85,1
  ```

---

## 📦 Dependencies

Make sure the following packages are installed:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow openpyxl matplotlib seaborn
```

---

## 🙋‍♀️ Team & Motivation

This project was completed as part of the **WiDS Datathon 2025** in collaboration with the Ann S. Bowers Women’s Brain Health Initiative.  
We are motivated by the goal of improving early detection and equitable diagnosis of ADHD, especially in girls who are historically underdiagnosed.

---

## 📫 Contact

If you have questions about this project or would like to collaborate, feel free to reach out via Kaggle or GitHub!
