import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from model_utils import get_predictions


def analyze_isotopologue_errors(test_df):
    """
    Group regression errors by isotopologue.
    
    Returns a dict keyed by isotopologue with MAE, RMSE, bias.
    """
    # Aggregate errors
    results = {}
    for iso in np.unique(test_df['iso']):
        if iso == "unknown":
            continue
        mask = test_df['iso'] == iso
        original_mae = mean_absolute_error(test_df['E_Ma_iso'][mask], test_df['E_IE_original'][mask])
        corrected_mae = mean_absolute_error(test_df['E_Ma_iso'][mask], test_df['E_IE_corrected'][mask])
        original_rmse = root_mean_squared_error(test_df['E_Ma_iso'][mask], test_df['E_IE_original'][mask])
        corrected_rmse = root_mean_squared_error(test_df['E_Ma_iso'][mask], test_df['E_IE_corrected'][mask])
        bias = np.mean(test_df['Original_error'][mask] - test_df['Corrected_error'][mask])
        results[iso] = {
            "Original MAE": original_mae,
            "Corrected MAE": corrected_mae,
            "Original RMSE": original_rmse,
            "Corrected RMSE": corrected_rmse,
            "Bias": bias,
            "Count": mask.sum(),
        }

    return results


def save_isotopologue_error_report(results, output_dir):
    """
    Save isotopologue error metrics (MAE, RMSE, Bias, Count) to CSV.
    """
    os.makedirs(os.path.join(output_dir, "CSVs"), exist_ok=True)
    df = pd.DataFrame(results).T
    df.index.name = "Isotopologue"
    df.to_csv(os.path.join(output_dir, "CSVs/isotopologue_errors.csv"))
    return df
