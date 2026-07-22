"""
Script for training and evaluating CatBoost models on molecular feature combinations.
Project: SimSJSAlert
"""

import itertools
import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==========================================
# CẤU HÌNH THAM SỐ CHUNG
# ==========================================
BASE_PREFIX = "SJS"
N_RUNS = 3  # Số lần chạy lặp lại để tính Mean ± SD

# Bật/tắt các đặc trưng (features) chủ động
FEATURE_TOGGLE = {
    "ChemBERTa": True,
    "RDKit": True,
    "MACCS": True,
    "ECFP": True,
    "Phychem": True,
    "Similarity": True,
}

print("=" * 80)
print("LOADING DATA FOR CATBOOST")
print("=" * 80)

available_features = {}

# 1. ChemBERTa
if FEATURE_TOGGLE.get("ChemBERTa", False):
    try:
        available_features["ChemBERTa"] = (
            np.load(f"{BASE_PREFIX}_x_train_ChemBERTa.npy"),
            np.load(f"{BASE_PREFIX}_x_test_ChemBERTa.npy"),
        )
    except FileNotFoundError:
        print("ChemBERTa files not found - skipped.")

# 2. RDKit
if FEATURE_TOGGLE.get("RDKit", False):
    try:
        available_features["RDKit"] = (
            pd.read_csv(f"{BASE_PREFIX}_x_train_rdkit.csv", index_col=0).values,
            pd.read_csv(f"{BASE_PREFIX}_x_test_rdkit.csv", index_col=0).values,
        )
    except FileNotFoundError:
        print("RDKit files not found - skipped.")

# 3. MACCS
if FEATURE_TOGGLE.get("MACCS", False):
    try:
        available_features["MACCS"] = (
            pd.read_csv(f"{BASE_PREFIX}_x_train_maccs.csv", index_col=0).values,
            pd.read_csv(f"{BASE_PREFIX}_x_test_maccs.csv", index_col=0).values,
        )
    except FileNotFoundError:
        print("MACCS files not found - skipped.")

# 4. ECFP
if FEATURE_TOGGLE.get("ECFP", False):
    try:
        available_features["ECFP"] = (
            pd.read_csv(f"{BASE_PREFIX}_x_train_ecfp.csv", index_col=0).values,
            pd.read_csv(f"{BASE_PREFIX}_x_test_ecfp.csv", index_col=0).values,
        )
    except FileNotFoundError:
        print("ECFP files not found - skipped.")

# 5. Phychem
if FEATURE_TOGGLE.get("Phychem", False):
    try:
        available_features["Phychem"] = (
            pd.read_csv(f"{BASE_PREFIX}_x_train_phychem.csv", index_col=0).values,
            pd.read_csv(f"{BASE_PREFIX}_x_test_phychem.csv", index_col=0).values,
        )
    except FileNotFoundError:
        print("Phychem files not found - skipped.")

# 6. Similarity Matrix
if FEATURE_TOGGLE.get("Similarity", False):
    try:
        available_features["Similarity"] = (
            np.load(f"{BASE_PREFIX}_x_train_similarity_matrix.npy"),
            np.load(f"{BASE_PREFIX}_x_test_similarity_matrix.npy"),
        )
    except FileNotFoundError:
        print("Similarity Matrix files not found - skipped.")

# Load Labels
y_train = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values.ravel()
y_test = pd.read_csv(f"{BASE_PREFIX}_y_test.csv", index_col=0).values.ravel()

# Sinh các kịch bản kết hợp đặc trưng (Đơn lẻ và Đôi một)
scenarios = {}
feature_names = list(available_features.keys())

for r in range(1, 3):
    for combo in itertools.combinations(feature_names, r):
        combo_name = "+".join(combo)
        train_list = [available_features[f][0] for f in combo]
        test_list = [available_features[f][1] for f in combo]
        scenarios[combo_name] = (
            np.concatenate(train_list, axis=1),
            np.concatenate(test_list, axis=1),
        )

print(f"\nGenerated {len(scenarios)} feature scenarios for CatBoost.")

# ==========================================
# VÒNG LẶP HUẤN LUYỆN VÀ ĐÁNH GIÁ CATBOOST
# ==========================================
results = []

for name, (x_train, x_test) in scenarios.items():
    print(f"\n{'='*80}\nScenario: {name} (Dims: {x_train.shape[1]})\n{'='*80}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
    x_test_scaled = scaler.transform(x_test.astype(np.float32))

    run_metrics = {
        "cb_mcc": [], "cb_acc": [], "cb_bacc": [], 
        "cb_auc": [], "cb_auprc": [], "cb_sens": [], "cb_spec": []
    }

    for run in range(N_RUNS):
        cb = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_seed=42 + run,
            auto_class_weights="Balanced",
            verbose=0,
            thread_count=-1,
        )
        cb.fit(x_train_scaled, y_train, eval_set=(x_test_scaled, y_test))

        cb_prob = cb.predict_proba(x_test_scaled)[:, 1]
        cb_pred = cb.predict(x_test_scaled)

        tn, fp, fn, tp = confusion_matrix(y_test, cb_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        run_metrics["cb_mcc"].append(matthews_corrcoef(y_test, cb_pred))
        run_metrics["cb_acc"].append(accuracy_score(y_test, cb_pred))
        run_metrics["cb_bacc"].append(balanced_accuracy_score(y_test, cb_pred))
        run_metrics["cb_auc"].append(roc_auc_score(y_test, cb_prob))
        run_metrics["cb_auprc"].append(average_precision_score(y_test, cb_prob))
        run_metrics["cb_sens"].append(sens)
        run_metrics["cb_spec"].append(spec)

    # Tính toán Mean và SD
    summary_row = {"Feature Set": name, "Dims": x_train.shape[1]}
    
    for metric_key, values in run_metrics.items():
        mean_val = np.mean(values)
        sd_val = np.std(values)
        
        metric_tag = metric_key.split("_")[1].upper().replace("AUC", "AUROC")
        summary_row[f"{metric_tag}-CB"] = f"{mean_val:.3f} ± {sd_val:.3f}"
        
        if metric_key == "cb_mcc":
            summary_row["CB-MCC-RAW"] = mean_val

    results.append(summary_row)

# ==========================================
# XUẤT KẾT QUẢ RA FILE CSV
# ==========================================
main_df = pd.DataFrame(results)
cb_cols = ["Feature Set", "Dims", "MCC-CB", "ACC-CB", "BACC-CB", "SENS-CB", "SPEC-CB", "AUROC-CB", "AUPRC-CB"]
cb_outputs_df = main_df[cb_cols]

output_filename = f"{BASE_PREFIX}_cb_metrics_outputs.csv"
cb_outputs_df.to_csv(output_filename, index=False)

print(f"\n[SUCCESS] CatBoost results successfully saved to {output_filename}")
