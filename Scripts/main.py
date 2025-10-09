import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import time

# ======================
# Config
# ======================
DATA_FILE = "Data/CO_CO2_combined.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EPOCHS = 100
LR = 1e-3
VAL_FRAC = 0.2
N_SPLITS = 5  # KFold splits
USE_SCHEDULER = True  # <-- Toggle LR scheduler on/off easily

ISO_WEIGHTS = {
    27: 3.0,
    28: 0.75,
    36: 2.0,
    37: 4.0,
    38: 4.0,
}

FEATURE_COLS = [
    "E_IE", "E_Ca_iso", "E_Ca_main", "E_Ma_main", "E_Ma_iso",
    "gtot", "J", "v",
    "AFGL_m1", "AFGL_m2", "AFGL_l2", "AFGL_m3", "AFGL_r",
    "hzb_v1", "hzb_v2", "hzb_l2", "hzb_v3",
    "Trove_v1", "Trove_v2", "Trove_v3", "Trove_coeff",
    "mu", "mu_ratio",
    "mu1", "mu2", "mu3", "mu_all",
    "mu1_ratio", "mu2_ratio", "mu3_ratio", "mu_all_ratio",
    "mass_c_12.0", "mass_c_13.003355",
    "mass_o_1_15.994915", "mass_o_1_16.999132", "mass_o_1_17.999161",
    "mass_o_2_15.994915", "mass_o_2_16.999132", "mass_o_2_17.999161",
    "e", "f", "Sym_Adp", "Sym_Ap", "Sym_A1", "Sym_A2",
]

SCALED_COLS = ["E_IE", "E_Ca_iso", "E_Ca_main", "E_Ma_main", "gtot", "J"]
TARGET_COL = "Error_IE"

# ======================
# Dataset
# ======================
class CorrectionDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, scaler=None, fit=False):
        X = df[feature_cols].copy()
        if fit:
            self.scaler = StandardScaler()
            X[SCALED_COLS] = self.scaler.fit_transform(X[SCALED_COLS])
        else:
            self.scaler = scaler
            X[SCALED_COLS] = self.scaler.transform(X[SCALED_COLS])

        self.X = torch.tensor(X.values.astype(np.float32))
        self.y = torch.tensor(df[target_col].values.astype(np.float32))
        self.mol_idx = torch.tensor(df["molecule_idx"].values.astype(np.int64))
        self.iso_idx = torch.tensor(df["iso_idx"].values.astype(np.int64))
        self.iso = df["iso"].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mol_idx[idx], self.iso_idx[idx], self.iso[idx]

# ======================
# Model (Hybrid Shared + Partial Heads)
# ======================
class CorrectionRegressor(nn.Module):
    def __init__(self, input_dim, n_molecules, n_isos, shared_dim=128, iso_head_dim=32):
        super().__init__()
        self.mol_embed = nn.Embedding(n_molecules, 8)
        self.iso_embed = nn.Embedding(n_isos, 8)

        total_dim = input_dim + 8 + 8
        self.trunk = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, shared_dim),
            nn.GELU(),
        )

        # Shared conditional trunk head
        self.shared_head = nn.Sequential(
            nn.Linear(shared_dim + 8, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Lightweight per-isotope adapters (fine-tuning layer)
        self.iso_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, iso_head_dim),
                nn.GELU(),
                nn.Linear(iso_head_dim, 1)
            ) for _ in range(n_isos)
        ])

        # Gating mechanism: learn per-isotope mixing coefficient (0–1)
        self.gate = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mol_idx, iso_idx):
        mol_emb = self.mol_embed(mol_idx)
        iso_emb = self.iso_embed(iso_idx)

        # Shared trunk representation
        z = torch.cat([x, mol_emb, iso_emb], dim=1)
        shared = self.trunk(z)

        # Shared conditional prediction
        shared_pred = self.shared_head(torch.cat([shared, iso_emb], dim=1))

        # Per-isotope specialized adapter prediction
        out = torch.zeros_like(shared_pred)
        for i, head in enumerate(self.iso_heads):
            mask = (iso_idx == i)
            if mask.any():
                out[mask] = head(shared[mask])

        # Gate determines how much to trust isotope-specific vs shared
        gate_val = self.gate(iso_emb)
        final_pred = gate_val * out + (1 - gate_val) * shared_pred

        return final_pred.squeeze(-1)

    def init_output_bias(self, bias_value):
        for module in reversed(list(self.trunk)):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.bias.fill_(float(bias_value))
                break


# ======================
# Adaptive Isotopologue Weighting
# ======================
class IsoWeightTracker:
    def __init__(self, base_weights, alpha=0.3):
        self.base_weights = base_weights.copy()
        self.alpha = alpha
        self.prev_mae = {}

    def update(self, iso_mae_dict):
        for iso, mae in iso_mae_dict.items():
            if iso in self.prev_mae:
                self.prev_mae[iso] = self.alpha * mae + (1 - self.alpha) * self.prev_mae[iso]
            else:
                self.prev_mae[iso] = mae

        mae_vals = np.array(list(self.prev_mae.values()))
        inv_mae = 1.0 / (mae_vals + 1e-6)
        inv_mae /= inv_mae.mean()
        for (iso, _), inv in zip(self.prev_mae.items(), inv_mae):
            self.base_weights[iso] = float(inv)

    def get(self):
        return self.base_weights

# ======================
# Training / Evaluation
# ======================
def train_epoch(model, loader, optimizer, iso_counts):
    model.train()
    total_loss = 0
    criterion = nn.HuberLoss(delta=0.01)

    for X, y, mol_idx, iso_idx, _ in loader:
        X, y, mol_idx, iso_idx = X.to(DEVICE), y.to(DEVICE), mol_idx.to(DEVICE), iso_idx.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X, mol_idx, iso_idx)
        sample_weights = torch.tensor([1.0 / iso_counts[i.item()] for i in iso_idx], device=DEVICE)
        loss = (criterion(preds, y) * sample_weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    criterion = nn.HuberLoss(delta=0.01)
    preds_all, y_all, iso_all = [], [], []

    with torch.no_grad():
        for X, y, mol_idx, iso_idx, iso in loader:
            X, y, mol_idx, iso_idx = X.to(DEVICE), y.to(DEVICE), mol_idx.to(DEVICE), iso_idx.to(DEVICE)
            preds = model(X, mol_idx, iso_idx)
            loss = criterion(preds, y)
            total_loss += loss.item() * len(y)
            preds_all.extend(preds.cpu().numpy())
            y_all.extend(y.cpu().numpy())
            iso_all.extend(iso)

    avg_loss = total_loss / len(loader.dataset)
    preds_array = np.array(preds_all)
    y_array = np.array(y_all)
    rmse = np.sqrt(np.mean((preds_array - y_array) ** 2))
    mae = np.mean(np.abs(preds_array - y_array))

    # Compute per-isotope MAE for adaptive weighting
    iso_mae = {}
    for i, iso in enumerate(iso_all):
        if iso not in iso_mae:
            iso_mae[iso] = []
        iso_mae[iso].append(abs(preds_array[i] - y_array[i]))
    iso_mae = {iso: np.mean(vals) for iso, vals in iso_mae.items()}

    return avg_loss, rmse, mae, iso_mae

# ======================
# Main with KFold
# ======================
def main():
    start_time = time.time()
    df = pd.read_csv(DATA_FILE)
    df = df[df["Error_IE"].abs() <= 1]
    df = df[df["iso"] != 636]
    df["iso_idx"] = df["iso"].astype("category").cat.codes

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results = []
    all_test_results = []

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")

        trainval_df = df.iloc[trainval_idx]
        test_df = df.iloc[test_idx]
        train_df, val_df = train_test_split(trainval_df, test_size=0.2, random_state=42)

        train_ds = CorrectionDataset(train_df, FEATURE_COLS, TARGET_COL, fit=True)
        val_ds = CorrectionDataset(val_df, FEATURE_COLS, TARGET_COL, scaler=train_ds.scaler)
        test_ds = CorrectionDataset(test_df, FEATURE_COLS, TARGET_COL, scaler=train_ds.scaler)

        iso_counts = train_df["iso_idx"].value_counts().to_dict()
        weights = train_df["iso_idx"].map(lambda i: 1.0 / iso_counts[i]).astype(float)
        def apply_iso_weight(row):
            return ISO_WEIGHTS.get(row["iso"], 1.0)
        weights *= train_df.apply(apply_iso_weight, axis=1)
        sampler = WeightedRandomSampler(weights.values, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        n_molecules = df["molecule_idx"].nunique()
        n_isos = df["iso_idx"].nunique()
        model = CorrectionRegressor(len(FEATURE_COLS), n_molecules, n_isos).to(DEVICE)
        model.init_output_bias(train_df[TARGET_COL].mean())

        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS) if USE_SCHEDULER else None

        iso_tracker = IsoWeightTracker(ISO_WEIGHTS)

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, iso_counts)
            val_loss, val_rmse, val_mae, val_iso_mae = evaluate(model, val_loader)
            iso_tracker.update(val_iso_mae)

            if scheduler:
                scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                      f"Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}")

        test_loss, test_rmse, test_mae, _ = evaluate(model, test_loader)
        fold_results.append((test_loss, test_rmse, test_mae))
        print(f"Fold {fold+1} Results | Loss: {test_loss:.6f} | RMSE: {test_rmse:.6f} | MAE: {test_mae:.6f}")

        model.eval()
        preds_all = []
        with torch.no_grad():
            for X, y, mol_idx, iso_idx, iso in test_loader:
                X, mol_idx, iso_idx = X.to(DEVICE), mol_idx.to(DEVICE), iso_idx.to(DEVICE)
                preds = model(X, mol_idx, iso_idx).cpu().numpy()
                preds_all.extend(preds)

        test_df_copy = test_df.copy()
        test_df_copy["NN_correction"] = preds_all
        test_df_copy["E_IE_corrected"] = test_df_copy["E_IE"] + test_df_copy["NN_correction"]
        test_df_copy["Original_error"] = test_df_copy["E_Ma_iso"] - test_df_copy["E_IE"]
        test_df_copy["Corrected_error"] = test_df_copy["E_Ma_iso"] - test_df_copy["E_IE_corrected"]
        test_df_copy["Original_abs_error"] = np.abs(test_df_copy["Original_error"])
        test_df_copy["Corrected_abs_error"] = np.abs(test_df_copy["Corrected_error"])
        all_test_results.append(test_df_copy)

    mean_loss = np.mean([r[0] for r in fold_results])
    mean_rmse = np.mean([r[1] for r in fold_results])
    mean_mae = np.mean([r[2] for r in fold_results])
    print(f"\n=== KFold {N_SPLITS}-fold Summary ===")
    print(f"Avg Loss: {mean_loss:.6f} | Avg RMSE: {mean_rmse:.6f} | Avg MAE: {mean_mae:.6f}")

    results_df = pd.concat(all_test_results, ignore_index=True)
    print("\nCO isotopologue improvements:")
    for iso in sorted(results_df["iso"].unique()):
        if len(str(iso)) == 2:
            sub = results_df[results_df["iso"] == iso]
            orig_mae = sub["Original_abs_error"].mean()
            corr_mae = sub["Corrected_abs_error"].mean()
            improvement = 100 * (orig_mae - corr_mae) / orig_mae
            print(f"Iso {iso}: {improvement:+.2f}%")

    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()
