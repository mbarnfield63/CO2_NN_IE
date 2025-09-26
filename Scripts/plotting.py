import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


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


def plot_predictions_vs_true(y_true, y_pred, output_dir, cv=False, all_preds_df=None):
    os.makedirs(os.path.join(output_dir, "Plots/Errors"), exist_ok=True)

    plt.figure(figsize=(6, 6))
    if cv == True:
        sns.scatterplot(data=all_preds_df, x="y_true", y="y_pred", hue="fold", palette="tab10", s=10, alpha=0.5)
        lims = [min(all_preds_df["y_true"].min(), all_preds_df["y_pred"].min()), max(all_preds_df["y_true"].max(), all_preds_df["y_pred"].max())]
    else:
        sns.scatterplot(data=None, x=y_true, y=y_pred, s=10, alpha=0.5, color="purple")
        lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]

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


def plot_iso_residuals_all(test_df, energy_col='E_Ma_iso', n_col=3, output_dir=None):
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
    last_ax = axes[-1, -1]
    axes[-1, -1].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    last_ax.legend(
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
            os.makedirs(os.path.join(output_dir, "Plots/Isotopologues"), exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"Plots/Isotopologues/{iso}_residuals.png"),
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


def plot_isotopologue_comparison(results, output_dir, figsize=(8, 6)):
    """
    Compare isotopologue errors across multiple metrics (e.g., MAE vs RMSE).

    Parameters
    ----------
    results : dict
        Output from analyze_isotopologue_errors
    output_dir : str
        Directory to save plots
    metrics : tuple
        Which metrics to compare
    figsize : tuple
        Size of the figure
    """
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
    plt.savefig(os.path.join(output_dir, "Plots/Errors/isotopologue_error_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
