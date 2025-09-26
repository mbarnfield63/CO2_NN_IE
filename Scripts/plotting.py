import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score

def plot_loss(train_losses, val_losses, output_dir):
    os.makedirs(os.path.join(output_dir, "Plots/Training"), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/Training/loss_plot.png"))
    plt.close()


def plot_predictions_vs_true(y_true, y_pred, output_dir, metrics=True, cv=False, all_preds_df=None):
    os.makedirs(os.path.join(output_dir, "Plots/Errors"), exist_ok=True)

    plt.figure(figsize=(6, 6))
    if cv == True:
        sns.scatterplot(data=all_preds_df, x="y_true", y="y_pred", hue="fold", palette="tab10", s=10, alpha=0.5)
        lims = [min(all_preds_df["y_true"].min(), all_preds_df["y_pred"].min()), max(all_preds_df["y_true"].max(), all_preds_df["y_pred"].max())]
        plt.legend(title="Fold No.",
                   title_fontsize=12,
                   fontsize=10,
                   loc='lower right')
        if metrics:
          r2 = r2_score(all_preds_df["y_true"], all_preds_df["y_pred"])
          rmse = np.sqrt(np.mean((all_preds_df["y_true"] - all_preds_df["y_pred"]) ** 2))
          plt.text(0.05, 0.95, f'$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}', transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top')

    else:
        sns.scatterplot(data=None, x=y_true, y=y_pred, s=10, alpha=0.5, color="purple")
        lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
        if metrics:
          r2 = r2_score(y_true, y_pred)
          rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
          plt.text(0.05, 0.95, f'$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}', transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top')

    sns.lineplot(x=lims, y=lims, linestyle='--', lw=2, color='black')  # y=x line   
    plt.xlabel("True Energy")
    plt.ylabel("Predicted Energy")
    plt.title("Predicted vs True Energy")
    plt.tight_layout()
    if cv == True:
        plt.savefig(os.path.join(output_dir, "Plots/Errors/pred_vs_true_cv.png"))
    else:
        plt.savefig(os.path.join(output_dir, "Plots/Errors/pred_vs_true.png"))
    plt.close()


def plot_iso_residuals_all(test_df, overall_pct_improvement, energy_col='E_Ma_iso', n_col=4, output_dir=None):
    """
    Plot energy distributions based on the original energy values for each isotopologue.
    """
    # Get unique isotopologues
    all_isos = sorted(test_df['iso'].unique())
   
    # Calculate grid dimensions
    n_isos = len(all_isos)
    n_rows = (n_isos + n_col - 1) // n_col
   
    # Get overall energy range for consistent x-axis limits
    energy_min = test_df[energy_col].min()
    energy_max = test_df[energy_col].max()
   
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_col, sharex=True, sharey=True, figsize=(5*n_col, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_col == 1:
        axes = axes.reshape(-1, 1)
   
    colors = ['blue', 'orange']
 
    for idx, iso in enumerate(all_isos):
        row = idx // n_col
        col = idx % n_col
        ax = axes[row, col]
        
        # Get mask for current isotopologue
        iso_mask = test_df['iso'] == iso
        
        ax.scatter(
            test_df.loc[iso_mask, energy_col],
            test_df["Original_error"][iso_mask],  # E_IE - E_Ma_iso
            s=10, alpha=0.5, color=colors[0], label='Original Error'
        )
        ax.scatter(
            test_df.loc[iso_mask, energy_col],
            test_df["Corrected_error"][iso_mask],  # E_IE_corrected - E_Ma_iso
            s=10, alpha=0.5, color=colors[1], label='NN Corrected'
        )
        ax.axhline(0, color='black', linestyle='--')
        
        # Calculate mean average error reduction percentage for this isotopologue
        mean_reduction = 100 * (test_df["Original_abs_error"][iso_mask].mean() - \
                                test_df["Corrected_abs_error"][iso_mask].mean()) / \
                                test_df["Original_abs_error"][iso_mask].mean()
              
        ax.text(0.05, 0.05, f'Iso: {iso}\nReduction: {mean_reduction:.2f}%',
                transform=ax.transAxes, fontsize=16, va='bottom')
       
        # Set consistent axis limits
        ax.set_xlim(energy_min, energy_max)
        if row == n_rows - 1:
            ax.set_xlabel('Energy (cm^-1)')
        ax.set_ylim(-0.15, 0.15)
        if col == 0:
            ax.set_ylabel('Residual (Obs - Calc)')
        ax.grid(True, alpha=0.3)
   
    # Add legend to the last row, last column axis
    penultimate_ax = axes[-1, -2]
    axes[-1, -1].axis('off')
    axes[-1, -2].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    penultimate_ax.legend(
        handles,
        labels,
        loc='upper center',
        fontsize=20,
        handlelength=2,
        handletextpad=1,
        markerscale=2,
        title='Residuals',
        title_fontsize=25
    )

    mae_reduction_text = f"Overall Error Improvement\n{overall_pct_improvement:.2f}%"
    penultimate_ax.text(0.5, 0.5, mae_reduction_text, transform=penultimate_ax.transAxes,
                        fontsize=20, va='top', ha='center')
   
    plt.tight_layout()
   
    if output_dir:
        os.makedirs(os.path.join(output_dir, "Plots/Isotopologues"), exist_ok=True)
        plt.savefig(os.path.join(output_dir, "Plots/Isotopologues/isotopologue_residuals.png"),
                   dpi=300, bbox_inches='tight')
    plt.close()


def plot_iso_residuals_individual(test_df, energy_col='E_Ma_iso', output_dir=None):
    # Get unique isotopologues
    all_isos = sorted(test_df['iso'].unique())
    colors = ['blue', 'orange']
    max_error = np.ceil(max(test_df["Original_abs_error"].abs().max(), test_df["Corrected_abs_error"].abs().max()) * 100) / 100

    for iso in all_isos:
        plt.figure(figsize=(6, 5))
        
        # Get mask for current isotopologue
        iso_mask = test_df['iso'] == iso
        
        plt.scatter(
            test_df.loc[iso_mask, energy_col],
            test_df["Original_error"][iso_mask],  # E_IE - E_Ma_iso
            s=5, alpha=0.5, color=colors[0], label='Original Error'
        )
        plt.scatter(
            test_df.loc[iso_mask, energy_col],
            test_df["Corrected_error"][iso_mask],  # E_IE_corrected - E_Ma_iso
            s=5, alpha=0.5, color=colors[1], label='NN Corrected'
        )

        plt.axhline(0, color='black', linestyle='--')
        
        # Calculate mean error reduction percentage for this isotopologue
        mean_reduction = 100 * (test_df["Original_abs_error"][iso_mask].mean() - test_df["Corrected_abs_error"][iso_mask].mean()) / test_df["Original_abs_error"][iso_mask].mean()
        
        plt.text(0.05, 0.95, f'Iso: {iso}\nReduction: {mean_reduction:.2f}%',
                 transform=plt.gca().transAxes, fontsize=12, va='top')
        
        plt.xlabel('Energy (cm^-1)')
        plt.ylabel('Residual (Obs - Calc)')
        plt.ylim(-max_error, max_error)
        plt.title(f'Isotopologue {iso} Residuals')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(os.path.join(output_dir, "Plots/Isotopologues/Individual"), exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"Plots/Isotopologues/Individual/{iso}_residuals.png"),
                        dpi=300, bbox_inches='tight')
        plt.close()



def plot_feature_importance(df, output_dir):
    os.makedirs(os.path.join(output_dir, "Plots/Features"), exist_ok=True)

    sorted_df = df.sort_values("importance", ascending=True)

    plt.figure(figsize=(12, 8))
    sorted_df.plot(kind="bar", legend=False, color="teal", y="importance", x="feature")
    plt.xticks(rotation=45, ha="right")
    plt.title("Permutation Feature Importance (RMSE Increase)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/Features/feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_mae_bars(results, output_dir, figsize=(8, 6)):
    isotopologues = sorted(results.keys())
    maes = ['Original MAE', 'Corrected MAE']
    x = np.arange(len(isotopologues))
    width = 0.35

    plt.figure(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(maes)))
    for i, metric in enumerate(maes):
        values = [results[iso][metric] for iso in isotopologues]
        plt.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)

    plt.xticks(x + width * (len(maes)-1) / 2, isotopologues)
    plt.ylabel("Error")
    plt.xlabel("Isotopologue (OCO notation)")
    plt.title("Isotopologue Error Comparison", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/Errors/mae_bars.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_bars(results, output_dir, figsize=(8, 6)):
    isotopologues = sorted(results.keys())
    metrics = ['Original MAE', 'Corrected MAE', 'Original RMSE', 'Corrected RMSE']
    x = np.arange(len(isotopologues))
    width = 0.2

    plt.figure(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    for i, metric in enumerate(metrics):
        values = [results[iso][metric] for iso in isotopologues]
        plt.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)

    plt.xticks(x + width * (len(metrics)-1) / 2, isotopologues)
    plt.ylabel("Error")
    plt.xlabel("Isotopologue (OCO notation)")
    plt.title("Isotopologue Error Comparison", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/Errors/mae_rmse_bars.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals_boxplot(test_df, output_dir=None):
    """
    Create two box plots (before and after correction) for all isotopologues.
    """
    all_isos = sorted(test_df['iso'].unique())
    data_original = []
    data_corrected = []

    for iso in all_isos:
        iso_mask = test_df['iso'] == iso
        data_original.append(test_df["Original_error"][iso_mask])
        data_corrected.append(test_df["Corrected_error"][iso_mask])

    fig, axes = plt.subplots(2, 1, figsize=(2.5*len(all_isos), 10), sharex=True)

    # Original residuals boxplot
    box1 = axes[0].boxplot(data_original, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', color='blue'),
                           medianprops=dict(color='red'))
    axes[0].axhline(0, color='black', linestyle='--')
    axes[0].set_ylabel('Residual (Obs - Calc)')
    axes[0].text(0.02, 0.02, 'Original IE Method Residuals', transform=axes[0].transAxes,
                 fontsize=22, fontweight='bold', va='bottom', ha='left')

    # Corrected residuals boxplot
    box2 = axes[1].boxplot(data_corrected, patch_artist=True,
                           boxprops=dict(facecolor='lightgreen', color='green'),
                           medianprops=dict(color='red'))
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set_ylabel('Residual (Obs - Calc)')
    axes[1].text(0.02, 0.02, 'Residuals after ML Correction', transform=axes[1].transAxes,
                 fontsize=22, fontweight='bold', va='bottom', ha='left')

    axes[1].set_xticks(range(1, len(all_isos)+1))
    axes[1].set_xticklabels(all_isos, rotation=45, ha='right', fontsize=16)
    axes[1].set_xlabel('Isotopologue')

    for ax in axes:
        ax.set_ylim(-0.1, 0.15)

    plt.tight_layout()

    if output_dir:
        os.makedirs(os.path.join(output_dir, "Plots/Isotopologues"), exist_ok=True)
        plt.savefig(os.path.join(output_dir, "Plots/Isotopologues/residuals_boxplot.png"),
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_hist_error_energy(test_df, energy_col='E_Ma_iso', output_dir=None):
    os.makedirs(os.path.join(output_dir, "Plots/Errors"), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Histogram of original errors
    axes[0].hist(test_df["Original_error"], bins=50, color='lightblue', edgecolor='blue', alpha=0.7)
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set_xlabel('Residual (Obs - Calc)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Original IE Method Residuals')

    # Histogram of corrected errors
    axes[1].hist(test_df["Corrected_error"], bins=50, color='lightgreen', edgecolor='green', alpha=0.7)
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_xlabel('Residual (Obs - Calc)')
    axes[1].set_title('Residuals after ML Correction')

    for ax in axes:
        ax.set_xlim(-0.1, 0.1)

    plt.subplots_adjust(wspace=0)
    if output_dir:
        plt.savefig(os.path.join(output_dir, "Plots/Errors/hist_error_energy.png"),
                    dpi=300, bbox_inches='tight')
    plt.close()
