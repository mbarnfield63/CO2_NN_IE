# model_utils.py
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from plotting import plot_feature_importance

# ===== Dataset
class CO2Dataset(Dataset):
    """
    Dataset for single-target energy regression with auxiliary IE (ionization energy) value.

    Args:
        df: DataFrame containing features, target, and IE columns
        feature_cols: list of feature column names
        target_col: name of the true energy column (regression target)
    """
    def __init__(self, df, feature_cols, target_col):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y[idx]),
        )


# ===== Data Loading
def load_data(
    path,
    feature_cols,
    scaled_cols,
    target_col,
    train_size=0.7,
    val_size=0.1,
    test_size=0.2,
    overlap_fraction=0.1,
    random_state=42,
    output_dir=None,
):
    """
    Load data for regression with optional sequential energy-based splitting.

    Returns:
        train_df, val_df, test_df, scaler
    """
    df = pd.read_csv(path)
    print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns.")

    # Validation checks
    required_cols = list(feature_cols) + [target_col]

    # Drop lines with NaN values in used columns
    df = df.dropna(subset=required_cols + (["iso"] if "iso" in df.columns else []))
    print(f"Dataset after dropping NaNs: {len(df)} samples.")

    # Store original energy values before any processing for plotting/splitting
    original_energy_col = "E_IE_original"
    df[original_energy_col] = df["E_IE"].copy()

    # Extract E_Ma_iso values before splitting to maintain index correspondence
    E_Ma_iso_values = df["E_Ma_iso"].values if "E_Ma_iso" in df.columns else None
    
    # Remove E_Ma_iso from dataframe to prevent data leakage
    if "E_Ma_iso" in df.columns:
        df = df.drop("E_Ma_iso", axis=1)

    # Simple random splits; keep relative sizes
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df, test_size=temp_size, random_state=random_state, shuffle=True
    )
    val_ratio = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.5
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_ratio), random_state=random_state + 1, shuffle=True
    )

    print(f"\nFinal split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Scale features (fit on train only to avoid leakage)
    scaler = StandardScaler()
    train_df.loc[:, scaled_cols] = scaler.fit_transform(train_df[scaled_cols])
    val_df.loc[:, scaled_cols] = scaler.transform(val_df[scaled_cols])
    test_df.loc[:, scaled_cols] = scaler.transform(test_df[scaled_cols])

    # Extract E_Ma_iso values corresponding to test set indices
    if E_Ma_iso_values is not None:
        E_Ma_iso_test = E_Ma_iso_values[test_df.index]
        print(f"Extracted E_Ma_iso_test array with {len(E_Ma_iso_test)} values.")
    else:
        E_Ma_iso_test = None
        print("Warning: E_Ma_iso column not found in dataset.")

    return train_df, val_df, test_df, scaler, E_Ma_iso_test


# ===== Model
class CO2EnergyRegressorSingle(nn.Module):
    def __init__(self, input_dim: int, dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(dropout),        
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),          
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.shared(x)


# ===== Loss function
def compute_loss(outputs, targets, **kwargs):
    """
    Standard MSE loss for regression.
    """
    return nn.MSELoss()(outputs, targets)


# ===== Train/Eval
def train(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_kwargs: dict | None = None,
):
    """
    Training loop for regression.
    Returns: average loss over the dataloader.
    """
    if loss_kwargs is None:
        loss_kwargs = {}

    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = compute_loss(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(1, len(dataloader))


def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    loss_kwargs: dict | None = None,
):
    """
    Evaluation loop for regression.
    Returns: average loss and RMSE over the dataloader.
    """
    if loss_kwargs is None:
        loss_kwargs = {}

    model.eval()
    total_loss = 0.0
    preds = []
    trues = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = compute_loss(outputs, y, **loss_kwargs)
            total_loss += float(loss.item())

            preds.append(outputs.view(-1).cpu().numpy())
            trues.append(y.view(-1).cpu().numpy())

    if preds:
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
    else:
        rmse = float("nan")

    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss, rmse, mae


# ===== Outputs
def get_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
):
    """
    Collects predictions and errors for analysis.
    Optionally inverse transforms predictions and targets to original scale.

    Returns:
        y_true (N,), y_pred (N,), abs_error (N,), signed_error (N,)
    """
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)

            y_pred_list.append(outputs.view(-1).cpu().numpy())
            y_true_list.append(y.view(-1).cpu().numpy())

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([])

    signed_error = y_pred - y_true
    abs_error = np.abs(signed_error)

    return y_true, y_pred, abs_error, signed_error


def get_feature_importance(
    model: nn.Module,
    dataloader,
    device: torch.device,
    feature_cols: list[str],
    output_dir: str | None = None,
    seed: int = 42,
    metric: str = "rmse",
):
    """
    Permutation feature importance for regression.
    Importance is measured as the increase in RMSE (or MSE/MAE) when permuting a feature.
    """
    assert metric in {"rmse", "mse", "mae"}
    model.eval()

    # Gather full dataset tensors from dataloader
    X_list, y_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_list.append(X_batch)
            y_list.append(y_batch)

    if not X_list:
        raise ValueError("Dataloader is empty; cannot compute feature importance.")

    X_full = torch.cat(X_list, dim=0).to(device)
    y_full = torch.cat(y_list, dim=0).to(device).view(-1, 1)

    # Baseline predictions
    with torch.no_grad():
        y_hat = model(X_full).view(-1, 1)

    def compute_metric(y_true_t, y_pred_t):
        diff = y_pred_t - y_true_t
        if metric == "rmse":
            return float(torch.sqrt(torch.mean(diff ** 2)).item())
        elif metric == "mse":
            return float(torch.mean(diff ** 2).item())
        else:  # mae
            return float(torch.mean(torch.abs(diff)).item())

    baseline = compute_metric(y_full, y_hat)

    # Set random seed for reproducibility of permutations
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    importance = {}

    with torch.no_grad():
        for j, feat in enumerate(feature_cols):
            X_perm = X_full.clone()
            idx = torch.randperm(X_perm.size(0), generator=g, device=device)
            X_perm[:, j] = X_perm[idx, j]

            y_perm = model(X_perm).view(-1, 1)
            score = compute_metric(y_full, y_perm)
            importance[feat] = score - baseline  # positive means worse (more important)

    # Present as DataFrame for downstream plotting
    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": [importance[feat] for feat in feature_cols]
    }).sort_values("importance", ascending=False)

    if output_dir:
        plot_feature_importance(df_imp, output_dir)

    return df_imp
