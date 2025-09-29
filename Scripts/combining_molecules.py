import os
import pandas as pd

# Unified feature superset (CO + CO2)
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

def align_features(df: pd.DataFrame, molecule: str) -> pd.DataFrame:
    """
    Align a dataframe (CO or CO2) to the unified FEATURE_COLS.
    Fill missing features with 0.
    Add molecule and molecule_idx columns.
    """
    # Add missing columns
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Restrict to feature set only (plus target, iso, etc.)
    keep_cols = FEATURE_COLS + ["Error_IE", "iso"]
    df = df.reindex(columns=keep_cols)

    # Add molecule label
    df["molecule"] = molecule
    return df


def load_molecule_data(co2_path: str, co_path: str, output_dir: str = "Data/Outputs") -> pd.DataFrame:
    """
    Load and combine CO2 + CO datasets into a unified dataframe.
    """
    if not os.path.exists(co2_path):
        raise FileNotFoundError(f"CO2 dataset not found: {co2_path}")
    if not os.path.exists(co_path):
        raise FileNotFoundError(f"CO dataset not found: {co_path}")

    print("Loading CO2 dataset...")
    co2_df = pd.read_csv(co2_path)
    co2_df = align_features(co2_df, "CO2")

    print("Loading CO dataset...")
    co_df = pd.read_csv(co_path)
    co_df = align_features(co_df, "CO")

    # Combine
    combined = pd.concat([co2_df, co_df], ignore_index=True)

    # Encode molecule_idx
    combined["molecule_idx"] = combined["molecule"].astype("category").cat.codes

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "CO_CO2_combined.csv")
    combined.to_csv(out_path, index=False)
    print(f"Unified dataset saved to {out_path} with {len(combined)} records")

    return combined


if __name__ == "__main__":
    # Example usage
    co2_file = "Data/CO2_minor_isos_ma.txt"
    co_file = "Data/CO_minor_isos_ma.txt"
    load_molecule_data(co2_file, co_file, output_dir="Data/")
