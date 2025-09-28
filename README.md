# AIML_PROJECT
Employee Attrition — Random Forest

# AIML_PROJECT — Employee Attrition & Performance Prediction

## 📌 Project Summary
This repository contains Task 1 for the AIML Recruitment Task: building Random Forest models to predict employee **Attrition** (classification) and **Performance Rating** (regression).  

All experiments, models, and outputs are saved in the `task1_improved_outputs/` folder.

---

## 📊 Dataset
- **File:** `Employee_Performance_Retention.csv`  
- **Rows:** 9000  
- **Columns:**  
  - Demographics: Age, Department  
  - Work-related: Years_of_Experience, Monthly_Working_Hours, Training_Hours_per_Year  
  - Satisfaction/History: Job_Satisfaction_Level, Promotion_in_Last_2_Years  
  - Targets:  
    - `Attrition` (Yes/No → classification)  
    - `Performance_Rating` (1–5 → regression)

---

## ⚙️ Methods Used
- Preprocessing:
  - Dropped `Employee_ID`
  - Encoded binary & categorical features (one-hot for Department, Job Satisfaction)
- Class imbalance handling:
  - Class weights
  - SMOTE oversampling
  - SMOTEENN (SMOTE + cleaning)
  - RandomizedSearchCV for hyperparameter tuning
  - Threshold tuning to maximize F1
- Models:
  - RandomForestClassifier (baseline + tuned)
  - RandomForestRegressor (extra credit)
  - XGBoost (baseline)

---

## 🏆 Results

### Attrition Classification
| Model                        | Accuracy | Precision | Recall | F1   | PR-AUC |
|-------------------------------|----------|-----------|--------|------|--------|
| RF (class_weight=balanced)    | 0.805    | 0.50      | 0.003  | 0.006| 0.198  |
| RF + SMOTE                    | 0.792    | 0.17      | 0.017  | 0.031| 0.202  |
| RF + SMOTEENN                 | 0.724    | 0.18      | 0.120  | 0.145| 0.193  |
| Tuned RF (SMOTE)              | 0.794    | 0.19      | 0.017  | 0.031| 0.202  |
| **Tuned RF + threshold=0.088**| **0.260**| **0.20**  | **0.935**| **0.33** | **0.202** |
| **XGBoost (baseline)**        | **0.688**| **0.23**  | **0.26** | **0.25** | **0.210** |

👉 **Final choice:** *Tuned Random Forest with threshold=0.088* — because high recall is important in HR (catching attritions).

---

### Performance Rating Regression
- RandomForestRegressor metrics:  
  - **MAE:** 0.91  
  - **RMSE:** 1.15  
  - **R²:** -0.046 (weak predictive power)

---

## 🔍 Feature Importance
Top features influencing attrition:  
1. Monthly_Working_Hours  
2. Training_Hours_per_Year  
3. Age  
4. Years_of_Experience  

(See `task1_improved_outputs/classifier_top15_importances.png`)

---

## 📂 Repository Structure
