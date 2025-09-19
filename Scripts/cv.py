import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import KFold

from model_utils import CO2Dataset, CO2EnergyRegressorSingle, train, evaluate, get_predictions
from plotting import plot_predictions_vs_true

# === Config
DATA_PATH = "Data/CO2_minor_isos_ma.txt"
OUTPUT_DIR = "Data/Outputs/CV"
FEATURE_COLS = [
    "E_IE", "E_Ca_iso", "E_Ca_main", "E_Ma_main", "gtot", "J",
    "AFGL_m1", "AFGL_m2", "AFGL_l2", "AFGL_r",
    "Trove_v1", "Trove_v2", "Trove_v3", "Trove_coeff",
    "mu1", "mu2", "mu3", "mu_all", "mu1_ratio", "mu2_ratio", "mu3_ratio", "mu_all_ratio",
    "mass_c_12.0", "mass_c_13.003355",
    "mass_o_1_15.994915", "mass_o_1_16.999132", "mass_o_1_17.999161",
    "mass_o_2_15.994915", "mass_o_2_16.999132", "mass_o_2_17.999161",
    "e", "f", "Sym_Adp", "Sym_Ap", "Sym_A1", "Sym_A2",
]
TARGET_COL = "Error_IE"

BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load full dataset
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
print(f"Loaded dataset with {len(df)} samples.")

# === Cross-validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_results = []
all_preds = []  # to collect per-sample predictions across folds

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n=== Fold {fold+1}/{K_FOLDS} ===")

    # Split into train/val
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    train_ds = CO2Dataset(train_df, FEATURE_COLS, TARGET_COL)
    val_ds = CO2Dataset(val_df, FEATURE_COLS, TARGET_COL)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Init model + optimizer
    model = CO2EnergyRegressorSingle(len(FEATURE_COLS)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, DEVICE)

    # Evaluate
    val_loss, val_rmse, val_mae = evaluate(model, val_loader, DEVICE)
    print(f"Fold {fold+1}:\n  Val RMSE: {val_rmse:.4f}\n  MAE: {val_mae:.4f}")
    fold_results.append({"fold": fold+1, "val_rmse": val_rmse, "val_mae": val_mae})

    # Collect predictions for this fold
    y_true, y_pred, _, _ = get_predictions(model, val_loader, DEVICE)
    fold_preds = pd.DataFrame({
        "fold": fold+1,
        "y_true": y_true,
        "y_pred": y_pred
    })
    all_preds.append(fold_preds)

# === Results
results_df = pd.DataFrame(fold_results)
mean_rmse = results_df["val_rmse"].mean()
std_rmse = results_df["val_rmse"].std()
mean_mae = results_df["val_mae"].mean()
std_mae = results_df["val_mae"].std()

print("\n=== Cross-validation summary ===")
print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"MAE: {mean_mae:.4f} ± {std_mae:.4f}")

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "cv_summary.txt"), "w") as f:
    f.write(f"Cross-validation RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
    f.write(f"Cross-validation MAE: {mean_mae:.4f} ± {std_mae:.4f}\n")

# Save combined predictions
all_preds_df = pd.concat(all_preds, ignore_index=True)
all_preds_df.to_csv(os.path.join(OUTPUT_DIR, "cv_predictions.csv"), index=False)

# Plot all predictions vs truth
plot_predictions_vs_true(y_true, y_pred, OUTPUT_DIR, cv=True, all_preds_df=all_preds_df)
print(f"\nResults and plots saved to {OUTPUT_DIR}")
