import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
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


def plot_predictions_vs_true(y_true, y_pred, output_dir, metrics=True):
    os.makedirs(os.path.join(output_dir, "Plots/Errors"), exist_ok=True)

    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=10, alpha=0.5, color="#440154")
    if metrics:
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        plt.text(0.05, 0.95, f'$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}', transform=plt.gca().transAxes,
                    verticalalignment='top')
    sns.lineplot(x=[-0.125, 0.125], y=[-0.125, 0.125], linestyle='--', lw=2, color='black')
    plt.xlabel("True Residual (Original IE)")
    #plt.xlim(-0.125, 0.125)
    plt.ylabel("Predicted Residual (ML Correction)")
    #plt.ylim(-0.125, 0.125)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/Errors/pred_vs_true.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions_vs_true_cv(all_preds_df, output_dir, metrics=True):
    os.makedirs(os.path.join(output_dir, "Plots/Errors"), exist_ok=True)

    plt.figure(figsize=(5, 5))
    sns.scatterplot(data=all_preds_df, x="y_true", y="y_pred", hue="fold", palette="tab10", s=10, alpha=0.5)
    lims = [min(all_preds_df["y_true"].min(), all_preds_df["y_pred"].min()),
            max(all_preds_df["y_true"].max(), all_preds_df["y_pred"].max())]
    plt.legend(title="Fold No.", loc='lower right')
    if metrics:
        r2 = r2_score(all_preds_df["y_true"], all_preds_df["y_pred"])
        rmse = np.sqrt(np.mean((all_preds_df["y_true"] - all_preds_df["y_pred"]) ** 2))
        plt.text(0.05, 0.95, f'$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}', transform=plt.gca().transAxes,
                    verticalalignment='top')
    sns.lineplot(x=lims, y=lims, linestyle='--', lw=2, color='black')
    plt.xlabel("True Residual (Original IE)")
    plt.ylabel("Predicted Residual (ML Correction)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/Errors/pred_vs_true_cv.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_iso_residuals_all(test_df, overall_pct_improvement, energy_col='E_Ma_iso', output_dir=None):
    """
    Plot energy distributions based on the original energy values for each isotopologue.
    Layout:
    X B C
    A B C
    
    Where X is legend, A=36, B=27,37, C=28,38
    """
    # Define the layout
    layout = {
        (0, 0): 'Legend',
        (0, 1): [27],
        (0, 2): [28],
        (1, 0): [36],
        (1, 1): [37],
        (1, 2): [38]
    }
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    colors = ['#21918c', '#440154']
    
    # Get energy range for each column
    col_energy_ranges = {}
    for col in range(3):
        isos_in_col = []
        for row in range(2):
            if (row, col) in layout and layout[(row, col)] != 'Legend':
                isos_in_col.extend(layout[(row, col)])
        
        if isos_in_col:
            col_mask = test_df['iso'].isin(isos_in_col)
            col_energy_ranges[col] = (
                test_df.loc[col_mask, energy_col].min(),
                test_df.loc[col_mask, energy_col].max()
            )
    
    # Plot each cell according to layout
    for (row, col), content in layout.items():
        ax = axes[row, col]
        
        if content == 'Legend':
            # Legend in top-left
            ax.axis('off')
            
            # Create dummy plots for legend
            dummy_ax = fig.add_subplot(111, frame_on=False)
            dummy_ax.scatter([], [], s=10, alpha=0.7, color=colors[0], marker='^',
                            label='Original IE Method')
            dummy_ax.scatter([], [], s=10, alpha=0.7, color=colors[1], marker='o',
                            label='IE + ML Correction')
            dummy_ax.axis('off')
            
            handles, labels = dummy_ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='center', fontsize=20,
                        handlelength=2, handletextpad=0.75, markerscale=5)
            
            mae_reduction_text = f"Overall Residuals Reduction\n{overall_pct_improvement:.2f}%"
            ax.text(0.5, 0.35, mae_reduction_text, transform=ax.transAxes,
                    fontsize=20, va='top', ha='center')
            
            dummy_ax.remove()
            continue
        
        # Plot isotopes
        for iso in content:
            iso_mask = test_df['iso'] == iso
            
            ax.scatter(
                test_df.loc[iso_mask, energy_col],
                test_df["Original_error"][iso_mask],
                s=10, alpha=0.7, color=colors[0], marker='^',
                label='Original IE Method' if iso == content[0] else '',
            )
            ax.scatter(
                test_df.loc[iso_mask, energy_col],
                test_df["Corrected_error"][iso_mask],
                s=10, alpha=0.7, color=colors[1], marker='o',
                label='IE + ML Correction' if iso == content[0] else '',
            )
            
            # Calculate mean reduction for this isotope
            mean_reduction = 100 * (test_df["Original_abs_error"][iso_mask].mean() - \
                                    test_df["Corrected_abs_error"][iso_mask].mean()) / \
                                    test_df["Original_abs_error"][iso_mask].mean()
            
            # Position text appropriately for single vs multiple isos
            y_pos = 0.95 if len(content) == 1 else 0.95 + (content.index(iso) * 0.15)
            ax.text(0.3, y_pos, f'Iso: {iso}\nReduction: {mean_reduction:.2f}%',
                    transform=ax.transAxes, fontsize=16, va='top')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.9)
        
        # x-axes
        if col in col_energy_ranges:
            ax.set_xlim(col_energy_ranges[col])
        
        if col == 0: ax.set_xlim(0, 50000)
        elif col == 1: ax.set_xlim(0, 9000)
        elif col == 2: ax.set_xlim(0, 45000)
        else: pass
        # Turn off x-tick labels on upper row
        if row == 0:
            ax.set_xticklabels([])

        # y-axis
        ylim_dict = {0: (-0.5, 0.5), 1: (-0.02, 0.02), 2: (-0.06, 0.06)}
        if col in ylim_dict:
            ax.set_ylim(ylim_dict[col])
        
        # Labels
        if row == 1:  # Bottom row
            ax.set_xlabel(r'MARVEL Energy cm$\mathregular{^{-1}}$')
        if col == 0:  # Left column
            ax.set_ylabel(r'Residual ($\it{Obs-Calc}$)')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.3)  # Gap between columns
    
    if output_dir:
        os.makedirs(os.path.join(output_dir, "Plots/Isotopologues"), exist_ok=True)
        plt.savefig(os.path.join(output_dir, "Plots/Isotopologues/isotopologue_residuals.png"),
                    dpi=300, bbox_inches='tight')
    plt.close()

def plot_iso_residuals_individual(test_df, energy_col='E_Ma_iso', output_dir=None):
    """
    Plot energy distributions based on the original energy values for each isotopologue, individually.
    """
    all_isos = sorted(test_df['iso'].unique())
    colors = ['#21918c', '#440154']
    max_error = np.ceil(max(test_df["Original_abs_error"].abs().max(), test_df["Corrected_abs_error"].abs().max()) * 100) / 100

    for iso in all_isos:
        plt.figure(figsize=(6, 5))
        iso_mask = test_df['iso'] == iso
        plt.scatter(
            test_df.loc[iso_mask, energy_col],
            test_df["Original_error"][iso_mask],
            s=10, alpha=0.7, color=colors[0], marker='^',
            label='Original IE Method'
        )
        plt.scatter(
            test_df.loc[iso_mask, energy_col],
            test_df["Corrected_error"][iso_mask],
            s=10, alpha=0.7, color=colors[1], marker='o',
            label='IE + ML Correction'
        )
        plt.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.9)

        mean_reduction = 100 * (test_df["Original_abs_error"][iso_mask].mean() - 
                                test_df["Corrected_abs_error"][iso_mask].mean()) / \
                                test_df["Original_abs_error"][iso_mask].mean()

        plt.text(0.05, 0.05, f'Iso: {iso}\nReduction: {mean_reduction:.2f}%',
                    transform=plt.gca().transAxes, fontsize=16, va='bottom')

        plt.xlabel(r'MARVEL Energy cm$\mathregular{^{-1}}$')
        plt.ylabel(r'Residual ($\it{Obs-Calc}$)')
        #plt.ylim(-max_error, max_error)
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
    n_features = len(sorted_df)

    fig, ax = plt.subplots(figsize=(15, 8))
    # Create bar plot
    ax.bar(range(n_features), sorted_df["importance"].values, color="teal")
    # Set x-ticks with better spacing
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(sorted_df["feature"].values, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Importance (MAE Increase)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Plots/Features/feature_importance.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_mae_bars(results, output_dir, figsize=(8, 6)):
    isotopologues = sorted(results.keys())
    maes = ['Original IE MAE', 'ML Corrected MAE']
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


def plot_metrics_bars(results, output_dir, figsize=(12, 5)):
    isotopologues = sorted(results.keys())
    maes = ['Original IE MAE', 'ML Corrected MAE']
    rmses = ['Original IE RMSE', 'ML Corrected RMSE']
    x = np.arange(len(isotopologues))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # MAE subplot
    for i, metric in enumerate(maes):
        values = [results[iso][metric] for iso in isotopologues]
        axes[0].bar(x + i * width, values, width, label=metric,
                    color="#21918c" if i == 0 else "#440154",
                    hatch="\\" if i == 0 else None)
    
    axes[0].set_xlabel("Isotopologue Number (OCO notation)")
    axes[0].set_xticks(x + width * (len(maes)-1) / 2)
    axes[0].set_xticklabels(isotopologues, rotation=45, ha='center')
    axes[0].set_ylabel("MAE")
    axes[0].set_ylim(0, 0.05)
    axes[0].legend(loc='upper left')
    axes[0].grid(axis="y")
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False)
    axes[0].tick_params(axis='y')

    # RMSE subplot
    for i, metric in enumerate(rmses):
        values = [results[iso][metric] for iso in isotopologues]
        axes[1].bar(x + i * width, values, width, label=metric,
                    color="#21918c" if i == 0 else "#440154",
                    hatch="\\" if i == 0 else None)
    
    axes[1].set_xlabel("Isotopologue Number (OCO notation)")
    axes[1].set_xticks(x + width * (len(rmses)-1) / 2)
    axes[1].set_xticklabels(isotopologues, rotation=45, ha='center')
    axes[1].set_ylabel("RMSE")
    axes[1].set_ylim(0, 0.08)
    axes[1].legend(loc='upper left')
    axes[1].grid(axis="y")
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False)
    axes[1].tick_params(axis='y')

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

    fig, axes = plt.subplots(1, 2, figsize=(5*len(all_isos), 10), sharey=True)

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

    # Calculate metrics
    orig_mae = np.mean(np.abs(test_df["Original_error"]))
    orig_rmse = np.sqrt(np.mean(test_df["Original_error"] ** 2))
    corr_mae = np.mean(np.abs(test_df["Corrected_error"]))
    corr_rmse = np.sqrt(np.mean(test_df["Corrected_error"] ** 2))

    # Histogram of original errors
    axes[0].hist(test_df["Original_error"], bins=50, color='#21918c', edgecolor='#21918c', alpha=0.7)
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set_xlabel(r'Residual ($\it{Obs-Calc}$)')
    axes[0].set_ylabel('Count')
    axes[0].text(
        0.05, 0.95,
        r"$\mathbf{{Original\ IE}}$"
        f"\nMAE: {orig_mae:.4f}\n"
        f"RMSE: {orig_rmse:.4f}",
        transform=axes[0].transAxes,
        fontsize=12,
        va='top',
        ha='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Histogram of corrected errors
    axes[1].hist(test_df["Corrected_error"], bins=50, color='#440154', edgecolor='#440154', alpha=0.7)
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_xlabel(r'Residual ($\it{Obs-Calc}$)')
    axes[1].text(
        0.05, 0.95,
        r"$\mathbf{{After\ ML\ Correction}}$"
        f"\nMAE: {corr_mae:.4f}\n"
        f"RMSE: {corr_rmse:.4f}",
        transform=axes[1].transAxes,
        fontsize=12,
        va='top',
        ha='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )


    axes[0].set_xlim(-0.12, 0.12)
    axes[1].set_xlim(-0.119, 0.12)    

    plt.subplots_adjust(wspace=0)
    if output_dir:
        plt.savefig(os.path.join(output_dir, "Plots/Errors/hist_error_energy.png"),
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_v0(test_df, output_dir=None):
    os.makedirs(os.path.join(output_dir, "Plots/Errors"), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=test_df, x='v0', y='Original_error', label='Original Error', color='blue', s=10, alpha=0.5)
    sns.scatterplot(data=test_df, x='v0', y='Corrected_error', label='NN Corrected', color='orange', s=10, alpha=0.5)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('v0 Quantum Number')
    plt.ylabel('Residual (Obs - Calc)')
    plt.title('Residuals vs v0 Quantum Number')
    plt.legend()
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "Plots/Errors/residuals_vs_v0.png"),
                    dpi=300, bbox_inches='tight')
    plt.close()
