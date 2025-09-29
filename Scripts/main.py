import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================
# Config
# ======================
DATA_FILE = "Data/CO_CO2_combined.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
EMBED_DIM = 4

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
TARGET_COL = "Error_IE"
SCALED_COLS = ["E_IE", "E_Ca_iso", "E_Ca_main", "E_Ma_main"]

# ======================
# Dataset
# ======================
class MoleculeDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, scaler=None,
                 fit=False, target_stats=None):
        X = df[feature_cols].copy()
        y = df[target_col].values.astype(np.float32)
        mol_idx = df["molecule_idx"].values.astype(np.int64)
        iso = df["iso"].values

        # Fit or apply feature scaler
        if fit:
            self.scaler = StandardScaler()
            X[SCALED_COLS] = self.scaler.fit_transform(X[SCALED_COLS])
        else:
            self.scaler = scaler
            X[SCALED_COLS] = self.scaler.transform(X[SCALED_COLS])

        self.X = torch.tensor(X.values.astype(np.float32))
        self.mol_idx = torch.tensor(mol_idx)
        self.iso = iso

        # Target normalization per molecule (this was correct)
        self.target_stats = target_stats
        y_norm = np.zeros_like(y)
        for mol in np.unique(mol_idx):
            mask = mol_idx == mol
            mu, sigma = self.target_stats[mol]
            y_norm[mask] = (y[mask] - mu) / sigma
        self.y = torch.tensor(y_norm)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mol_idx[idx], self.iso[idx]

# ======================
# Model
# ======================
class MoleculeRegressor(nn.Module):
    def __init__(self, input_dim, n_molecules, embed_dim=4, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(n_molecules, embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            # Single shared prediction head
            nn.Linear(128, 1)
        )

    def forward(self, x, mol_idx):
        emb = self.embed(mol_idx)
        z = torch.cat([x, emb], dim=1)
        out = self.trunk(z).squeeze(-1)
        return out

# ======================
# Training / Evaluation
# ======================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y, mol_idx, _ in loader:
        X, y, mol_idx = X.to(DEVICE), y.to(DEVICE), mol_idx.to(DEVICE)
        optimizer.zero_grad()
        
        # Simple forward pass - no balanced loss
        preds = model(X, mol_idx)
        loss = criterion(preds, y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, target_stats):
    model.eval()
    preds_all, y_all, mols, isos = [], [], [], []
    with torch.no_grad():
        for X, y, mol_idx, iso in loader:
            X, y, mol_idx = X.to(DEVICE), y.to(DEVICE), mol_idx.to(DEVICE)
            preds = model(X, mol_idx)

            preds_np = preds.cpu().numpy()
            y_np = y.cpu().numpy()
            mol_idx_np = mol_idx.cpu().numpy()

            # Denormalize per molecule
            for m in np.unique(mol_idx_np):
                mask = mol_idx_np == m
                mu, sigma = target_stats[m]
                preds_np[mask] = preds_np[mask] * sigma + mu
                y_np[mask] = y_np[mask] * sigma + mu

            preds_all.append(preds_np)
            y_all.append(y_np)
            mols.extend(mol_idx_np)
            isos.extend(iso)

    return (
        np.concatenate(y_all),
        np.concatenate(preds_all),
        np.array(mols),
        np.array(isos),
    )

# ======================
# Main
# ======================
def main():
    df = pd.read_csv(DATA_FILE)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Per-molecule target stats (this was correct)
    target_stats = {}
    for mol, sub in train_df.groupby("molecule_idx"):
        mu = sub[TARGET_COL].mean()
        sigma = sub[TARGET_COL].std() if sub[TARGET_COL].std() > 0 else 1.0
        target_stats[mol] = (mu, sigma)

    # Datasets
    train_ds = MoleculeDataset(train_df, FEATURE_COLS, TARGET_COL, fit=True, target_stats=target_stats)
    val_ds = MoleculeDataset(val_df, FEATURE_COLS, TARGET_COL, scaler=train_ds.scaler, target_stats=target_stats)
    test_ds = MoleculeDataset(test_df, FEATURE_COLS, TARGET_COL, scaler=train_ds.scaler, target_stats=target_stats)

    # Oversampling CO (assumes CO = molecule_idx == 1)
    weights = np.ones(len(train_df))
    co_idx = train_df["molecule_idx"].max()  # assumes CO is the higher index
    weights[train_df["molecule_idx"].values == co_idx] = 5.0
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MoleculeRegressor(len(FEATURE_COLS), df["molecule_idx"].nunique(), EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        tl = train_epoch(model, train_loader, optimizer, criterion)
        y_val, p_val, mols_val, _ = evaluate(model, val_loader, target_stats)
        vl = np.mean((y_val - p_val) ** 2)
        train_losses.append(tl)
        val_losses.append(vl)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {tl:.6f} | Val Loss: {vl:.6f}")

    # Plot curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Training Curve")
    plt.show()

    # Test evaluation
    y_test, p_test, mols, isos = evaluate(model, test_loader, target_stats)
    df_out = pd.DataFrame({"y_true": y_test, "y_pred": p_test, "molecule": mols, "iso": isos})
    inv_map = dict(enumerate(df["molecule"].astype("category").cat.categories))
    df_out["molecule"] = df_out["molecule"].map(inv_map)

    for mol in df_out["molecule"].unique():
        sub = df_out[df_out["molecule"] == mol]
        mae = np.mean(np.abs(sub["y_pred"] - sub["y_true"]))
        print(f"{mol} Test MAE: {mae:.6f}")

    print("\nPercentage improvements per isotopologue:")
    for iso, sub in df_out.groupby("iso"):
        baseline_mae = np.mean(np.abs(sub["y_true"]))  # naive baseline
        model_mae = np.mean(np.abs(sub["y_pred"] - sub["y_true"]))
        improvement = 100 * (baseline_mae - model_mae) / baseline_mae
        print(f"{iso}: {improvement:+.2f}%")

if __name__ == "__main__":
    main()
