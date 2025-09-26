# cv.py  -- robust cross-validation with fold-specific scaling and bias init
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from model_utils import (
    CO2Dataset,
    CO2EnergyRegressorSingle,
    evaluate,
    get_predictions,
    train_and_return_debug,
)
from plotting import plot_predictions_vs_true

# === Config (match main.py feature list exactly)
DATA_PATH = "Data/CO2_minor_isos_ma.txt"
OUTPUT_DIR = "Data/Outputs/"
FEATURE_COLS = [
    "E_IE", "E_Ca_iso", "E_Ca_main", "E_Ma_main", "gtot", "J",
    "AFGL_m1", "AFGL_m2", "AFGL_l2", "AFGL_m3", "AFGL_r",
    "hzb_v1", "hzb_v2", "hzb_l2", "hzb_v3",
    "Trove_v1", "Trove_v2", "Trove_v3", "Trove_coeff",
    "mu1", "mu2", "mu3", "mu_all", "mu1_ratio", "mu2_ratio", "mu3_ratio", "mu_all_ratio",
    "mass_c_12.0", "mass_c_13.003355",
    "mass_o_1_15.994915", "mass_o_1_16.999132", "mass_o_1_17.999161",
    "mass_o_2_15.994915", "mass_o_2_16.999132", "mass_o_2_17.999161",
    "e", "f", "Sym_Adp", "Sym_Ap", "Sym_A1", "Sym_A2",
]
SCALED_COLS = ["E_IE", "E_Ca_iso", "E_Ca_main", "E_Ma_main", "gtot", "J"]
TARGET_COL = "Error_IE"

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "CSVs"), exist_ok=True)

# Load raw dataframe
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
print(f"Loaded dataset with {len(df)} samples.")

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
fold_results = []
all_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"\n=== Fold {fold+1}/{K_FOLDS} ===")
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # Fit scaler on train only
    scaler = StandardScaler()
    train_df.loc[:, SCALED_COLS] = scaler.fit_transform(train_df[SCALED_COLS])
    val_df.loc[:, SCALED_COLS] = scaler.transform(val_df[SCALED_COLS])

    # Prepare dataloaders
    train_ds = CO2Dataset(train_df, FEATURE_COLS, TARGET_COL)
    val_ds = CO2Dataset(val_df, FEATURE_COLS, TARGET_COL)
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

    # Model (match main: no weight_decay unless explicitly wanted)
    model = CO2EnergyRegressorSingle(len(FEATURE_COLS), dropout=0.3).to(DEVICE)
    # Initialize final bias to train-target mean for fold
    train_targets = train_df[TARGET_COL].values.astype(np.float32)
    mean_target = float(np.mean(train_targets))
    model.init_output_bias(mean_target)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # no weight_decay to match main run

    # Print initial quick diagnostics (train-sample preds)
    with torch.no_grad():
        # take first up to 10 training rows for a quick check
        small_train = train_df.head(min(10, len(train_df)))
        small_X = torch.from_numpy(small_train[FEATURE_COLS].values.astype(np.float32))
        init_train_preds = model(small_X.to(DEVICE)).view(-1).cpu().numpy()
    print("Initial sample train preds (first up to 10):", np.round(init_train_preds, 6))
    print("Train-target mean (used as init bias):", mean_target)

    # Training loop with debug
    collapsed_flag = False
    for epoch in range(EPOCHS):
        (
            train_loss,
            param_norm_before,
            grad_norm_before,
            param_norm_after,
            grad_norm_after,
        ) = train_and_return_debug(model, train_loader, optimizer, DEVICE)

        val_loss, val_rmse, val_mae = evaluate(model, val_loader, DEVICE)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}"
            )
            print(
                f"  Param norm before: {param_norm_before:.6f}, after: {param_norm_after:.6f}; "
                f"Grad norm before: {grad_norm_before:.6f}, after: {grad_norm_after:.6f}"
            )

        # Fast sanity check on training predictions after epoch 1: did the model collapse to nearly-constant?
        if epoch == 0:
            # compute preds on a small subset of training (first 50)
            small_n = min(50, len(train_df))
            small_X_full = torch.from_numpy(train_df[FEATURE_COLS].values[:small_n].astype(np.float32))
            with torch.no_grad():
                preds_small = model(small_X_full.to(DEVICE)).view(-1).cpu().numpy()
            std_preds = float(np.std(preds_small))
            mean_preds = float(np.mean(preds_small))
            print(f"  [Sanity] After 1st epoch: train preds mean={mean_preds:.6f}, std={std_preds:.8f}")
            # if std extremely tiny, warn (possible collapse)
            if std_preds < 1e-6:
                collapsed_flag = True
                print("  WARNING: very small prediction std on training subset after 1 epoch (possible collapse).")
                print("           Norms and grads printed above — consider lowering dropout, removing weight decay, or checking target scaling.")

    # Final evaluation
    val_loss, val_rmse, val_mae = evaluate(model, val_loader, DEVICE)
    print(f"Fold {fold+1} final | Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}")
    fold_results.append({"fold": fold + 1, "val_rmse": val_rmse, "val_mae": val_mae})

    # Collect predictions for this fold (all val samples)
    y_true, y_pred, _, _ = get_predictions(model, val_loader, DEVICE)

    # Print first 10 pairs to inspect constancy
    print("First 10 (y_true, y_pred):")
    for a, b in zip(y_true[:10], y_pred[:10]):
        print(f"  {a:.6f} -> {b:.6f}")

    fold_preds = pd.DataFrame({"fold": fold + 1, "y_true": y_true, "y_pred": y_pred})
    all_preds.append(fold_preds)

# Summary
results_df = pd.DataFrame(fold_results)
print("\n=== Cross-validation summary ===")
print(results_df)
print(f"RMSE: {results_df['val_rmse'].mean():.6f} ± {results_df['val_rmse'].std():.6f}")
print(f"MAE: {results_df['val_mae'].mean():.6f} ± {results_df['val_mae'].std():.6f}")

all_preds_df = pd.concat(all_preds, ignore_index=True)
all_preds_df.to_csv(os.path.join(OUTPUT_DIR, "CSVs/cv_predictions.csv"), index=False)

plot_predictions_vs_true(None, None, OUTPUT_DIR, cv=True, all_preds_df=all_preds_df)
print(f"\nResults and plots saved to {OUTPUT_DIR}")
