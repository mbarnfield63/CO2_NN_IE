import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import torch
from torch.utils.data import DataLoader
from torch import optim

from model_utils import *
from analysis import *
from plotting import *

# === Plotting parameters
thesis_params = {"xtick.minor.visible": True,
                    "xtick.major.pad":5,
                    "xtick.direction":"in",
                    "xtick.top":True,
                    "ytick.minor.visible": True,
                    "ytick.direction":"in",
                    "ytick.right":True,
                    "font.family":"DejaVu Sans",
                    "font.size":14.0,
                    "lines.linewidth":2,
                    "legend.frameon":False,
                    "legend.labelspacing":0,
                    "legend.borderpad":0.5,
                }
sns.set_theme(style='ticks', rc=thesis_params)
mpl.rcParams.update(thesis_params)

# === Setup
start_time = time.time()
print("Time start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# === Config
DATA_PATH = "Data/CO_CO2_combined.csv"
BATCH_SIZE = 512
EPOCHS = 100
LR = 5e-4
VAL_FRAC = 0.2
N_SPLITS = 5
SEEDS = [0]
USE_SCHEDULER = True

ISO_WEIGHTS = {
    27: 7.0,
    28: 1.0,
    36: 2.0,
    37: 7.0,
    38: 1.0,
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

TARGET_COL = "Error_IE"

# ======================
# Dataset
# ======================
class CorrectionDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, scaler=None, fit=False):
        X = df[feature_cols].copy()
        if fit:
            self.scaler = StandardScaler()
            X[:] = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X[:] = self.scaler.transform(X)

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
    def __init__(self, input_dim, n_molecules, n_isos, shared_dim=128, iso_head_dim=64):
        super().__init__()
        self.mol_embed = nn.Embedding(n_molecules, 8)
        self.iso_embed = nn.Embedding(n_isos, 8)

        total_dim = input_dim + 8 + 8
        self.trunk = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, shared_dim),
            nn.LayerNorm(shared_dim),
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


def get_predictions(model, loader, device):
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for X, y, mol_idx, iso_idx, _ in loader:
            X, mol_idx, iso_idx = X.to(device), mol_idx.to(device), iso_idx.to(device)
            outputs = model(X, mol_idx, iso_idx)
            y_pred_list.append(outputs.cpu().numpy())
            y_true_list.append(y.cpu().numpy())

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([])
    signed_error = y_pred - y_true
    abs_error = np.abs(signed_error)

    return y_true, y_pred, abs_error, signed_error

# ======================
# Main with Multi-Seed Averaging
# ======================
def main():
    start_time = time.time()

    all_seed_results = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n==============================")
        print(f" Running seed {seed}")
        print(f"==============================")

        # Set all random seeds for reproducibility
        import os, random
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        df = pd.read_csv(DATA_FILE)
        df = df[df["Error_IE"].abs() <= 1]
        df["iso_idx"] = df["iso"].astype("category").cat.codes
        
        # If not iso or molecule columns, convert to float
        for col in FEATURE_COLS:
            df[col] = df[col].astype(float)

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_results = []
        all_test_results = []

        for fold, (trainval_idx, test_idx) in enumerate(kf.split(df)):
            print(f"\n=== Fold {fold+1}/{N_SPLITS} (Seed {seed}) ===")

            trainval_df = df.iloc[trainval_idx]
            test_df = df.iloc[test_idx]
            train_df, val_df = train_test_split(trainval_df, test_size=0.2, random_state=seed)

            train_ds = CorrectionDataset(train_df, FEATURE_COLS, TARGET_COL, fit=True)
            val_ds = CorrectionDataset(val_df, FEATURE_COLS, TARGET_COL, scaler=train_ds.scaler)
            test_ds = CorrectionDataset(test_df, FEATURE_COLS, TARGET_COL, scaler=train_ds.scaler)

            iso_counts = train_df["iso_idx"].value_counts().to_dict()
            weights = train_df["iso_idx"].map(lambda i: 1.0 / iso_counts[i]).astype(float)
            def apply_iso_weight(row):
                return ISO_WEIGHTS.get(row["iso"], 1.0)
            weights *= train_df.apply(apply_iso_weight, axis=1)

            g = torch.Generator()
            g.manual_seed(seed)
            sampler = WeightedRandomSampler(weights.values, num_samples=len(weights), replacement=True, generator=g)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            n_molecules = df["molecule_idx"].nunique()
            n_isos = df["iso_idx"].nunique()
            model = CorrectionRegressor(len(FEATURE_COLS), n_molecules, n_isos).to(DEVICE)
            model.init_output_bias(train_df[TARGET_COL].mean())

            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS) if USE_SCHEDULER else None
            iso_tracker = IsoWeightTracker(ISO_WEIGHTS)

            for epoch in range(EPOCHS):
                train_loss = train_epoch(model, train_loader, optimizer, iso_counts)
                val_loss, val_rmse, val_mae, val_iso_mae = evaluate(model, val_loader)
                iso_tracker.update(val_iso_mae)
                if scheduler: scheduler.step()
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
                            f"Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f}")

            test_loss, test_rmse, test_mae, _ = evaluate(model, test_loader)
            fold_results.append((test_loss, test_rmse, test_mae))
            print(f"Fold {fold+1} (Seed {seed}) | RMSE: {test_rmse:.6f} | MAE: {test_mae:.6f}")

            # collect predictions
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

            # Save final fold model and create plots for last seed
            if seed_idx == len(SEEDS) - 1 and fold == N_SPLITS - 1:
                final_model = model
                final_train_losses = train_loss
                final_val_losses = val_loss
                final_test_df = test_df_copy
                final_test_loader = test_loader

        # Aggregate results for this seed
        results_df = pd.concat(all_test_results, ignore_index=True)
        # Filter for CO isotopologues only (2 digits)
        co_results_df = results_df[results_df["iso"].apply(lambda x: len(str(x)) == 2)].copy()
        seed_summary = {}
        print(f"\nSeed {seed} - CO Isotopologue Summary:")
        for iso in sorted(co_results_df["iso"].unique()):
            sub = co_results_df[co_results_df["iso"] == iso]
            orig_mae = sub["Original_abs_error"].mean()
            corr_mae = sub["Corrected_abs_error"].mean()
            improvement = 100 * (orig_mae - corr_mae) / orig_mae
            seed_summary[iso] = improvement
            print(f"  Iso {iso}: {improvement:+.2f}%")
        all_seed_results.append(seed_summary)

    # ======================
    # Final Summary Across Seeds
    # ======================
    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY (CO ISOTOPOLOGUES):")
    print("=" * 60)

    all_isos = sorted({iso for d in all_seed_results for iso in d.keys()})
    for iso in all_isos:
        vals = [d[iso] for d in all_seed_results if iso in d]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"Iso {iso}: {mean_val:+.2f}% ± {std_val:.2f}%")

    # ======================
    # Final Plots (Last Seed, CO Only)
    # ======================
    print("\n" + "=" * 60)
    print("GENERATING PLOTS (FINAL SEED, CO ISOTOPOLOGUES ONLY):")
    print("=" * 60)

    # Save final model
    os.makedirs("Models", exist_ok=True)
    torch.save(final_model.state_dict(), "Models/co_correction_model.pt")
    print("Model saved to Models/co_correction_model.pt")

    # Filter for CO isotopologues
    final_test_df_co = final_test_df[final_test_df["iso"].apply(lambda x: len(str(x)) == 2)].copy()

    # Get predictions for plotting
    y_true_final, y_pred_final, _, _ = get_predictions(final_model, final_test_loader, DEVICE)

    # Filter predictions for CO isotopologues
    co_mask = final_test_df["iso"].apply(lambda x: len(str(x)) == 2).values
    y_true_co = y_true_final[co_mask]
    y_pred_co = y_pred_final[co_mask]

    print("\nPlotting results...")
    plot_loss(final_train_losses, final_val_losses, OUTPUT_DIR)
    plot_predictions_vs_true(y_true_co, y_pred_co, OUTPUT_DIR)

    # Isotopologue analysis
    print("\nAnalyzing CO isotopologue-specific errors...")
    iso_results = analyze_isotopologue_errors(final_test_df_co)
    iso_df = save_isotopologue_error_report(iso_results, OUTPUT_DIR)
    print(iso_df)

    # Isotopologue plots
    plot_mae_bars(iso_results, OUTPUT_DIR)
    plot_metrics_bars(iso_results, OUTPUT_DIR)

    # Calculate overall improvement for CO isotopologues
    overall_pct_improvement = 100 * (final_test_df_co['Original_abs_error'].mean() - 
                                        final_test_df_co['Corrected_abs_error'].mean()) / \
                                        final_test_df_co['Original_abs_error'].mean()
    final_test_df_co['Error_reduction_pct'] = 100 * (final_test_df_co['Original_abs_error'] - 
                                                        final_test_df_co['Corrected_abs_error']) / \
                                                        final_test_df_co['Original_abs_error']

    print(f"\nCO Isotopologues - Original MAE: {final_test_df_co['Original_abs_error'].mean():.6f}")
    print(f"CO Isotopologues - Corrected MAE: {final_test_df_co['Corrected_abs_error'].mean():.6f}")
    print(f"CO Isotopologues - Overall MAE Improvement: {overall_pct_improvement:.2f}%")

    improved_samples = (final_test_df_co['Error_reduction_pct'] > 0).sum()
    print(f"Samples with improvement: {improved_samples}/{len(final_test_df_co)} "
            f"({100*improved_samples/len(final_test_df_co):.2f}%)")

    # Residual plots for CO isotopologues
    plot_iso_residuals_all(final_test_df_co, overall_pct_improvement, output_dir=OUTPUT_DIR)
    plot_iso_residuals_individual(final_test_df_co, output_dir=OUTPUT_DIR)
    plot_residuals_boxplot(final_test_df_co, output_dir=OUTPUT_DIR)
    plot_hist_error_energy(final_test_df_co, energy_col='E_Ma_iso', output_dir=OUTPUT_DIR)

    # Save final results
    os.makedirs(os.path.join(OUTPUT_DIR, "CSVs"), exist_ok=True)
    final_test_df_co.to_csv(os.path.join(OUTPUT_DIR, "CSVs/test_predictions_co.csv"), index=False)

    elapsed = time.time() - start_time
    print(f"\nTotal time for {len(SEEDS)} seeds: {elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()
