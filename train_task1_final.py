# train_task1_improved.py
"""
Improved Task 1 pipeline for Random Forest (classification + regression).
Drop this file into the same folder as Employee_Performance_Retention.csv and run:
    python train_task1_improved.py

Outputs saved to ./task1_improved_outputs/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")   # prevents tkinter GUI warnings on Windows
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from scipy.stats import randint
from collections import Counter

# --------- CONFIG ----------
DATA_PATH = "Employee_Performance_Retention.csv"   # must be in same folder
OUT_DIR = "task1_improved_outputs"
RANDOM_STATE = 42
N_ITER_SEARCH = 12   # keep small for speed; increase if you have more time
os.makedirs(OUT_DIR, exist_ok=True)

# --------- UTIL FUNCTIONS ----------
def preprocess(df):
    df = df.copy()
    # Drop ID if present
    if "Employee_ID" in df.columns:
        df = df.drop(columns=["Employee_ID"])
    # Map binary
    if "Promotion_in_Last_2_Years" in df.columns:
        df["Promotion_in_Last_2_Years"] = df["Promotion_in_Last_2_Years"].map({"Yes":1, "No":0})
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Yes":1, "No":0})
    # One-hot encode Department and Job_Satisfaction_Level (if present)
    cat_cols = [c for c in ["Department", "Job_Satisfaction_Level"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df

def print_class_results(y_true, y_pred, y_proba=None, title="Model"):
    print(f"\n--- {title} ---")
    print("Accuracy:", round(accuracy_score(y_true, y_pred),4))
    print("Precision:", round(precision_score(y_true, y_pred, zero_division=0),4))
    print("Recall:", round(recall_score(y_true, y_pred, zero_division=0),4))
    print("F1:", round(f1_score(y_true, y_pred, zero_division=0),4))
    if y_proba is not None:
        try:
            print("ROC AUC:", round(roc_auc_score(y_true, y_proba),4))
            print("Average Precision (PR AUC):", round(average_precision_score(y_true, y_proba),4))
        except Exception:
            pass
    print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=["No","Yes"]))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

def save_feature_importances(features, importances, fname_csv, fname_png, topn=15, title="Feature importances"):
    fi = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv(fname_csv, index=False)
    top = fi.head(topn).iloc[::-1]
    plt.figure(figsize=(8,6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname_png)
    plt.close()
    return fi

# --------- LOAD ----------
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------- PREPROCESS ----------
df_proc = preprocess(df)
print("After preprocess:", df_proc.shape)

# --------- Prepare classification data (Attrition) ----------
if "Attrition" not in df_proc.columns:
    raise RuntimeError("Attrition column missing after preprocessing.")

X = df_proc.drop(columns=["Attrition"])
y = df_proc["Attrition"]

print("\nClass distribution (original):", Counter(y))

# Split once (we will use same test set for experiments)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- Baseline 1: RandomForest with class_weight='balanced' ----------
print("\n>>> Baseline: RandomForest (class_weight='balanced')")
clf_baseline = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1)
clf_baseline.fit(X_train, y_train)
probs_base = clf_baseline.predict_proba(X_test)[:,1]
pred_base = clf_baseline.predict(X_test)
print_class_results(y_test, pred_base, probs_base, title="RF class_weight='balanced'")

# Save baseline
joblib.dump(clf_baseline, os.path.join(OUT_DIR, "rf_baseline_classweight.joblib"))

# ---------- Approach A: SMOTE oversampling ----------
print("\n>>> SMOTE oversampling on training set")
sm = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print("After SMOTE train distribution:", Counter(y_train_sm))

clf_sm = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
clf_sm.fit(X_train_sm, y_train_sm)
probs_sm = clf_sm.predict_proba(X_test)[:,1]
pred_sm = clf_sm.predict(X_test)
print_class_results(y_test, pred_sm, probs_sm, title="RF trained on SMOTE")

# ---------- Approach B: SMOTEENN (SMOTE + Edited Nearest Neighbours) ----------
print("\n>>> SMOTEENN (combine SMOTE + cleaning)")
smenn = SMOTEENN(random_state=RANDOM_STATE)
X_train_smenn, y_train_smenn = smenn.fit_resample(X_train, y_train)
print("After SMOTEENN train distribution:", Counter(y_train_smenn))

clf_smenn = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
clf_smenn.fit(X_train_smenn, y_train_smenn)
probs_smenn = clf_smenn.predict_proba(X_test)[:,1]
pred_smenn = clf_smenn.predict(X_test)
print_class_results(y_test, pred_smenn, probs_smenn, title="RF trained on SMOTEENN")

# ---------- Approach C: RandomizedSearchCV on SMOTE data (tuning) ----------
print("\n>>> RandomizedSearchCV (tuning) on SMOTE data — this may take some minutes")
param_dist = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(3, 30),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None, 0.3, 0.6],
    "bootstrap": [True, False]
}
rfc = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rand_search = RandomizedSearchCV(
    estimator=rfc,
    param_distributions=param_dist,
    n_iter=N_ITER_SEARCH,
    scoring="f1",
    cv=3,
    random_state=RANDOM_STATE,
    verbose=1,
    n_jobs=-1
)
rand_search.fit(X_train_sm, y_train_sm)   # tuned on SMOTE data
best_clf = rand_search.best_estimator_
print("Best params:", rand_search.best_params_)

# Evaluate tuned model
probs_tuned = best_clf.predict_proba(X_test)[:,1]
pred_tuned = best_clf.predict(X_test)
print_class_results(y_test, pred_tuned, probs_tuned, title="Tuned RF (trained on SMOTE)")

# Save tuned classifier and feature importances
joblib.dump(best_clf, os.path.join(OUT_DIR, "rf_tuned_on_smote.joblib"))
fi = save_feature_importances(X.columns, best_clf.feature_importances_,
                              os.path.join(OUT_DIR, "classifier_feature_importances.csv"),
                              os.path.join(OUT_DIR, "classifier_top15_importances.png"),
                              title="Top Feature Importances (Tuned RF on SMOTE)")

# ROC plot for tuned
fpr, tpr, _ = roc_curve(y_test, probs_tuned)
plt.figure()
plt.plot(fpr, tpr, label="Tuned RF")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned RF")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_tuned_rf.png"))
plt.close()

# PR curve (precision-recall) for tuned
ap = average_precision_score(y_test, probs_tuned)
prec, rec, th = precision_recall_curve(y_test, probs_tuned)
plt.figure()
plt.plot(rec, prec, label=f"AP={ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Tuned RF")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curve_tuned_rf.png"))
plt.close()

# ---------- Threshold tuning (choose probability cutoff that maximizes F1) ----------
print("\n>>> Threshold tuning (maximize F1) for tuned RF")
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, probs_tuned)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-12)
best_idx = np.nanargmax(f1_scores)
# thresholds length is len(f1_scores)-1, handle edge
if best_idx >= len(thresholds):
    best_thresh = 0.5
else:
    best_thresh = thresholds[best_idx]
print("Best threshold (max F1):", best_thresh)
pred_thresh = (probs_tuned >= best_thresh).astype(int)
print_class_results(y_test, pred_thresh, probs_tuned, title=f"Tuned RF (threshold={best_thresh:.3f})")

# Save thresholded predictions / small report
report = {
    "best_threshold": float(best_thresh),
    "tuned_rf_f1_at_best_threshold": float(f1_scores[best_idx]),
    "tuned_rf_ap": float(ap)
}
pd.Series(report).to_csv(os.path.join(OUT_DIR, "tuned_rf_threshold_report.csv"), index=True)

# Save final tuned model and threshold for deployment/report
import json
final_model_path = os.path.join(OUT_DIR, "rf_tuned_final.joblib")
joblib.dump(best_clf, final_model_path)
# Save the threshold alongside the model
with open(os.path.join(OUT_DIR, "rf_tuned_final_metadata.json"), "w") as f:
    json.dump({"threshold": float(best_thresh)}, f)

# Save test set predictions (prob, pred_at_threshold, true label)
test_out = X_test.copy()
test_out["true_attrition"] = y_test.values
test_out["prob_attrition"] = probs_tuned
test_out["pred_attrition_thresh"] = pred_thresh
test_out.to_csv(os.path.join(OUT_DIR, "test_predictions_with_threshold.csv"), index=False)
print("Saved final model, threshold metadata and test predictions.")

# ---------- Optional: Try XGBoost (if installed) ----------
try:
    from xgboost import XGBClassifier
    print("\n>>> Trying XGBoost with scale_pos_weight")
    scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_train, y_train)
    probs_xgb = xgb.predict_proba(X_test)[:,1]
    pred_xgb = xgb.predict(X_test)
    print_class_results(y_test, pred_xgb, probs_xgb, title="XGBoost (scale_pos_weight)")
    joblib.dump(xgb, os.path.join(OUT_DIR, "xgb_baseline.joblib"))
except Exception as e:
    print("XGBoost not available or failed:", e)

# ---------- REGRESSION: Performance_Rating (extra credit) ----------
if "Performance_Rating" in df_proc.columns:
    print("\n>>> REGRESSION: RandomForestRegressor for Performance_Rating")
    df_reg = df_proc.copy()
    X_reg = df_reg.drop(columns=["Performance_Rating"])
    y_reg = df_reg["Performance_Rating"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE)
    rfr = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rfr.fit(Xr_train, yr_train)
    yr_pred = rfr.predict(Xr_test)
    mae = mean_absolute_error(yr_test, yr_pred)
    mse = mean_squared_error(yr_test, yr_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(yr_test, yr_pred)
    print("MAE:", round(mae,4), "RMSE:", round(rmse,4), "R2:", round(r2,4))
    joblib.dump(rfr, os.path.join(OUT_DIR, "rf_regressor_performance.joblib"))
    # save feature importances for regressor too
    fi_reg = save_feature_importances(X_reg.columns, rfr.feature_importances_,
                                     os.path.join(OUT_DIR, "regressor_feature_importances.csv"),
                                     os.path.join(OUT_DIR, "regressor_top15_importances.png"),
                                     title="Top Feature Importances - Regressor")
    # actual vs predicted plot
    plt.figure(figsize=(6,6))
    plt.scatter(yr_test, yr_pred, alpha=0.4)
    plt.xlabel("Actual Performance Rating")
    plt.ylabel("Predicted Performance Rating")
    plt.title("Actual vs Predicted - Performance Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "regression_actual_vs_predicted.png"))
    plt.close()
else:
    print("Performance_Rating column not found; skipping regression.")

print("\nAll done. Outputs saved inside:", OUT_DIR)
print("Files:", os.listdir(OUT_DIR))
