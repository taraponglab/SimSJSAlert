"""
Script for training and evaluating PyTorch BiGRU models on molecular feature combinations.
Project: SimSJSAlert
"""

import itertools
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# CẤU HÌNH THAM SỐ CHUNG
# ==========================================
BASE_PREFIX = "SJS"
N_RUNS = 3  # Số lần chạy lặp lại để tính Mean ± SD

# Bật/tắt các đặc trưng chủ động
FEATURE_TOGGLE = {
    "ChemBERTa": True,
    "RDKit": True,
    "MACCS": True,
    "ECFP": True,
    "Phychem": True,
    "Similarity": True,
}

print("=" * 80)
print("LOADING DATA FOR BiGRU")
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

n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

# Sinh các kịch bản kết hợp đặc trưng
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

print(f"\nGenerated {len(scenarios)} feature scenarios for BiGRU.")


# ==========================================
# ĐỊNH NGHĨA KIẾN TRÚC MẠNG BiGRU
# ==========================================
class BiGRUMolecularClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(BiGRUMolecularClassifier, self).__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        logits = self.fc(out)
        return logits


# ==========================================
# VÒNG LẶP HUẤN LUYỆN VÀ ĐÁNH GIÁ BiGRU
# ==========================================
results = []

for name, (x_train, x_test) in scenarios.items():
    print(f"\n{'='*80}\nScenario: {name} (Dims: {x_train.shape[1]})\n{'='*80}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
    x_test_scaled = scaler.transform(x_test.astype(np.float32))

    run_metrics = {
        "gru_mcc": [], "gru_acc": [], "gru_bacc": [], 
        "gru_auc": [], "gru_auprc": [], "gru_sens": [], "gru_spec": []
    }

    for run in range(N_RUNS):
        torch.manual_seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + run)

        x_train_3d = np.expand_dims(x_train_scaled, axis=1)
        x_test_3d = np.expand_dims(x_test_scaled, axis=1)

        train_dataset = TensorDataset(
            torch.tensor(x_train_3d, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        test_tensor = torch.tensor(x_test_3d, dtype=torch.float32).to(device)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        input_dim = x_train_scaled.shape[1]
        bigru_model = BiGRUMolecularClassifier(input_dim=input_dim).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([scale_pos_weight]).to(device))
        optimizer = optim.Adam(bigru_model.parameters(), lr=0.001, weight_decay=1e-4)

        # Huấn luyện mô hình qua các epochs
        bigru_model.train()
        for epoch in range(15):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = bigru_model(batch_x).squeeze(1)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Đánh giá trên tập test
        bigru_model.eval()
        with torch.no_grad():
            bigru_logits = bigru_model(test_tensor).squeeze(1)
            bigru_prob = torch.sigmoid(bigru_logits).cpu().numpy()
            bigru_pred = (bigru_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, bigru_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        run_metrics["gru_mcc"].append(matthews_corrcoef(y_test, bigru_pred))
        run_metrics["gru_acc"].append(accuracy_score(y_test, bigru_pred))
        run_metrics["gru_bacc"].append(balanced_accuracy_score(y_test, bigru_pred))
        run_metrics["gru_auc"].append(roc_auc_score(y_test, bigru_prob))
        run_metrics["gru_auprc"].append(average_precision_score(y_test, bigru_prob))
        run_metrics["gru_sens"].append(sens)
        run_metrics["gru_spec"].append(spec)

    # Tính toán Mean và SD
    summary_row = {"Feature Set": name, "Dims": x_train.shape[1]}
    
    for metric_key, values in run_metrics.items():
        mean_val = np.mean(values)
        sd_val = np.std(values)
        
        metric_tag = metric_key.split("_")[1].upper().replace("AUC", "AUROC")
        summary_row[f"{metric_tag}-GRU"] = f"{mean_val:.3f} ± {sd_val:.3f}"
        
        if metric_key == "gru_mcc":
            summary_row["GRU-MCC-RAW"] = mean_val

    results.append(summary_row)

# ==========================================
# XUẤT KẾT QUẢ RA FILE CSV
# ==========================================
main_df = pd.DataFrame(results)
gru_cols = ["Feature Set", "Dims", "MCC-GRU", "ACC-GRU", "BACC-GRU", "SENS-GRU", "SPEC-GRU", "AUROC-GRU", "AUPRC-GRU"]
gru_outputs_df = main_df[gru_cols]

output_filename = f"{BASE_PREFIX}_bigru_metrics_outputs.csv"
gru_outputs_df.to_csv(output_filename, index=False)

print(f"\n[SUCCESS] BiGRU results successfully saved to {output_filename}")
