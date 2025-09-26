import numpy as np
import os
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
from torch import optim

from model_utils import *
from analysis import *
from plotting import *

# === Setup
start_time = time.time()
print("Time start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# === Config
DATA_PATH = "Data/CO2_minor_isos_ma.txt"
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "Data/Outputs"

# === Columns
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

MA_COL = "E_Ma_iso"

# === Data
train_df, val_df, test_df, scaler, E_Ma_iso_test = load_data(
    DATA_PATH,
    FEATURE_COLS,
    SCALED_COLS,
    target_col=TARGET_COL,
    output_dir=OUTPUT_DIR,
)

train_ds = CO2Dataset(train_df, FEATURE_COLS, TARGET_COL)
val_ds = CO2Dataset(val_df, FEATURE_COLS, TARGET_COL)
test_ds = CO2Dataset(test_df, FEATURE_COLS, TARGET_COL)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === Model
model = CO2EnergyRegressorSingle(len(FEATURE_COLS)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel created with {len(FEATURE_COLS)} input features.")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# === Train
print("\n" + "=" * 60)
print("MODEL TRAINING:")
print("=" * 60)
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, DEVICE)
    val_loss, val_rmse, val_mae = evaluate(model, val_loader, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1:2d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val RMSE: {val_rmse:.4f} | "
        f"Val MAE: {val_mae:.4f}"
    )

print("\n" + "=" * 60)
print("MODEL PERFORMANCE:")
print("=" * 60)

# === Test evaluation
test_loss, test_rmse, test_mae = evaluate(model, test_loader, DEVICE)
print(f"\nTest Loss: {test_loss:.4f}:\n  Test RMSE: {test_rmse:.4f}\n  Test MAE: {test_mae:.4f}")

# Save model
torch.save(model.state_dict(), "Models/co2_ie_model.pt")
print("Model saved to Models/co2_ie_model.pt")

# === Predictions
y_true, y_pred, abs_err, signed_err = get_predictions(model, test_loader, DEVICE)

# Post-processing test_df
post_processing(test_df, scaler, SCALED_COLS, y_pred, E_Ma_iso_test, OUTPUT_DIR)

# Isotopologue analysis
print("\nAnalyzing isotopologue-specific errors...")
iso_results = analyze_isotopologue_errors(test_df)
iso_df = save_isotopologue_error_report(iso_results, OUTPUT_DIR)
print(iso_df)

# === Plots
print("\nPlotting results...")
plot_loss(train_losses, val_losses, OUTPUT_DIR)
plot_predictions_vs_true(y_true, y_pred, OUTPUT_DIR)

# Isotopologue plot
plot_isotopologue_comparison(iso_results, OUTPUT_DIR)

# Feature importance
print("\nCalculating feature importance...")
feature_importance_df = get_feature_importance(model, test_loader, DEVICE, FEATURE_COLS, OUTPUT_DIR)
feature_importance_df.to_csv(os.path.join(OUTPUT_DIR, "CSVs/feature_importance.csv"))

# Isotopologue energy distributions and residuals
plot_iso_residuals_all(test_df, output_dir=OUTPUT_DIR)
plot_iso_residuals_individual(test_df, output_dir=OUTPUT_DIR)

# Print results summary
print(f"\nOriginal MAE: {test_df['Original_abs_error'].mean():.6f}")
print(f"Corrected MAE: {test_df['Corrected_abs_error'].mean():.6f}")
overall_pct_improvement = 100 * (test_df['Original_abs_error'].mean() - test_df['Corrected_abs_error'].mean()) / test_df['Original_abs_error'].mean()
print(f"Overall MAE Improvement: {overall_pct_improvement:.2f}%")

improved_samples = (test_df['Error_reduction_pct'] > 0).sum()
print(f"Samples with improvement: {improved_samples}/{len(test_df)} ({100*improved_samples/len(test_df):.2f}%)")

# === Final summary
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal training time: {elapsed_time:.2f} seconds")
print(f"  Time per epoch: {elapsed_time/EPOCHS:.2f} seconds")
print("\nAll outputs saved to:", OUTPUT_DIR)
