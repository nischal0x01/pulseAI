"""
Evaluation utilities for blood pressure prediction models.
Includes metrics calculation, visualization, and attention analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


def calculate_comprehensive_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics for blood pressure prediction.
    
    Args:
        y_true: True blood pressure values
        y_pred: Predicted blood pressure values
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays to ensure 1D for metric calculations
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Pearson correlation coefficient
    pearson_r, pearson_p = pearsonr(y_true_flat, y_pred_flat)
    
    # Mean percentage error
    mpe = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
    
    # Standard deviation of error
    errors = y_true_flat - y_pred_flat
    std_error = np.std(errors)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'MPE': mpe,
        'STD_Error': std_error,
        'Mean_Error': np.mean(errors),
        'Min_Error': np.min(errors),
        'Max_Error': np.max(errors)
    }
    
    return metrics


def print_metrics(metrics, dataset_name="Test"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Set Evaluation Metrics")
    print(f"{'='*60}")
    print(f"  📊 MAE (Mean Absolute Error):     {metrics['MAE']:>8.2f} mmHg")
    print(f"  📊 RMSE (Root Mean Squared Error): {metrics['RMSE']:>8.2f} mmHg")
    print(f"  📊 R² Score:                       {metrics['R2']:>8.4f}")
    print(f"  📊 Pearson Correlation:            {metrics['Pearson_r']:>8.4f} (p={metrics['Pearson_p']:.2e})")
    print(f"  📊 Mean Percentage Error:          {metrics['MPE']:>8.2f}%")
    print(f"  📊 Error STD:                      {metrics['STD_Error']:>8.2f} mmHg")
    print(f"  📊 Mean Error (bias):              {metrics['Mean_Error']:>8.2f} mmHg")
    print(f"{'='*60}")
    
    # Check if MAE target is met
    if metrics['MAE'] < 10.0:
        print(f"✅ SUCCESS: MAE < 10 mmHg target achieved!")
    else:
        print(f"⚠️  Target: MAE < 10 mmHg (current: {metrics['MAE']:.2f} mmHg)")
    print()


def plot_predictions_vs_actual(y_true, y_pred, dataset_name="Test", save_path=None):
    """
    Create scatter plot of predicted vs actual blood pressure.
    
    Args:
        y_true: True blood pressure values
        y_pred: Predicted blood pressure values
        dataset_name: Name of dataset (for title)
        save_path: Optional path to save figure
    """
    # Flatten arrays to ensure 1D
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.6, edgecolors='k', s=50, label='Predictions')
    
    # Perfect prediction line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics for title
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    pearson_r, _ = pearsonr(y_true_flat, y_pred_flat)
    
    ax.set_xlabel('Actual SBP (mmHg)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted SBP (mmHg)', fontsize=12, fontweight='bold')
    
    # Determine if this is SBP or DBP based on dataset_name
    bp_type = "SBP" if "SBP" in dataset_name or "sbp" in dataset_name.lower() else "Blood Pressure"
    if "DBP" in dataset_name or "dbp" in dataset_name.lower():
        bp_type = "DBP"
        ax.set_xlabel('Actual DBP (mmHg)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted DBP (mmHg)', fontsize=12, fontweight='bold')
    
    ax.set_title(f'{dataset_name}: Predicted vs Actual {bp_type}\n' +
                f'MAE={mae:.2f} mmHg, R²={r2:.4f}, Pearson r={pearson_r:.4f}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add diagonal reference bands (±10 mmHg)
    ax.fill_between([min_val, max_val], 
                    [min_val - 10, max_val - 10],
                    [min_val + 10, max_val + 10],
                    alpha=0.2, color='green', label='±10 mmHg band')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig


def plot_error_distribution(y_true, y_pred, dataset_name="Test", save_path=None):
    """
    Plot the distribution of prediction errors.
    
    Args:
        y_true: True blood pressure values
        y_pred: Predicted blood pressure values
        dataset_name: Name of dataset
        save_path: Optional path to save figure
    """
    # Flatten arrays to ensure 1D for plotting
    errors = (y_true - y_pred).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(np.mean(errors), color='green', linestyle='--', 
                    linewidth=2, label=f'Mean Error: {np.mean(errors):.2f}')
    axes[0].set_xlabel('Prediction Error (mmHg)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{dataset_name}: Error Distribution', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(errors, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].axhline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_ylabel('Prediction Error (mmHg)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{dataset_name}: Error Box Plot', 
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig


def visualize_attention_weights(attention_weights, sample_indices=None, num_samples=5, 
                                 save_path=None):
    """
    Visualize PAT-based attention weights over time.
    
    Args:
        attention_weights: Attention weights array (samples, timesteps, 1)
        sample_indices: Optional specific sample indices to visualize
        num_samples: Number of random samples to visualize if indices not provided
        save_path: Optional path to save figure
    """
    if sample_indices is None:
        # Select random samples
        sample_indices = np.random.choice(len(attention_weights), 
                                         size=min(num_samples, len(attention_weights)), 
                                         replace=False)
    
    fig, axes = plt.subplots(len(sample_indices), 1, 
                            figsize=(12, 3*len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]
    
    for idx, sample_idx in enumerate(sample_indices):
        weights = attention_weights[sample_idx].flatten()
        timesteps = np.arange(len(weights))
        
        axes[idx].plot(timesteps, weights, linewidth=2, color='darkblue')
        axes[idx].fill_between(timesteps, 0, weights, alpha=0.3, color='skyblue')
        axes[idx].set_xlabel('Time Step', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Attention Weight', fontsize=10, fontweight='bold')
        axes[idx].set_title(f'PAT-based Attention Weights - Sample {sample_idx}', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        # Highlight peak attention
        peak_idx = np.argmax(weights)
        axes[idx].axvline(peak_idx, color='red', linestyle='--', 
                         label=f'Peak at t={peak_idx}')
        axes[idx].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to: {save_path}")
    
    plt.show()
    return fig


def analyze_attention_statistics(attention_weights):
    """
    Analyze statistical properties of attention weights.
    
    Args:
        attention_weights: Attention weights array (samples, timesteps, 1)
        
    Returns:
        Dictionary of statistics
    """
    weights = attention_weights.reshape(attention_weights.shape[0], -1)
    
    stats = {
        'mean_entropy': -np.mean(np.sum(weights * np.log(weights + 1e-10), axis=1)),
        'mean_max_weight': np.mean(np.max(weights, axis=1)),
        'mean_peak_position': np.mean(np.argmax(weights, axis=1)),
        'std_peak_position': np.std(np.argmax(weights, axis=1)),
        'mean_weight_concentration': np.mean(np.sum(weights > 0.1, axis=1) / weights.shape[1])
    }
    
    print(f"\n{'='*60}")
    print("Attention Mechanism Analysis")
    print(f"{'='*60}")
    print(f"  🔍 Mean Entropy:              {stats['mean_entropy']:.4f}")
    print(f"  🔍 Mean Max Weight:           {stats['mean_max_weight']:.4f}")
    print(f"  🔍 Mean Peak Position:        {stats['mean_peak_position']:.2f}")
    print(f"  🔍 STD Peak Position:         {stats['std_peak_position']:.2f}")
    print(f"  🔍 Weight Concentration (>0.1): {stats['mean_weight_concentration']:.2%}")
    print(f"{'='*60}\n")
    
    return stats


def comprehensive_evaluation(model, X_test, y_test, dataset_name="Test", 
                             visualize_attention=True, save_dir=None):
    """
    Perform comprehensive model evaluation with all metrics and visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (can be dict with 'sbp' and 'dbp' keys or single array)
        dataset_name: Name for plots
        visualize_attention: Whether to extract and visualize attention
        save_dir: Directory to save plots (optional)
        
    Returns:
        Dictionary containing predictions, metrics, and attention weights (if available)
    """
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE MODEL EVALUATION - {dataset_name} Set")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Check if y_test is a dictionary (dual output) or single array
    is_dual_output = isinstance(y_test, dict) and 'sbp' in y_test and 'dbp' in y_test
    
    # Get predictions
    try:
        predictions = model.predict(X_test, verbose=0)
        
        # Handle different output formats
        if isinstance(predictions, list):
            if len(predictions) == 3:  # [SBP, DBP, Attention]
                y_pred_sbp, y_pred_dbp, attention_weights = predictions
                results['attention_weights'] = attention_weights
            elif len(predictions) == 2 and is_dual_output:  # [SBP, DBP]
                y_pred_sbp, y_pred_dbp = predictions
                attention_weights = None
            elif len(predictions) == 2:  # [Pred, Attention]
                y_pred_sbp = predictions[0]
                y_pred_dbp = None
                attention_weights = predictions[1]
                results['attention_weights'] = attention_weights
            else:
                y_pred_sbp = predictions[0] if len(predictions) > 0 else predictions
                y_pred_dbp = None
                attention_weights = None
        else:
            y_pred_sbp = predictions
            y_pred_dbp = None
            attention_weights = None
    except:
        predictions = model.predict(X_test, verbose=0)
        y_pred_sbp = predictions
        y_pred_dbp = None
        attention_weights = None
    
    # Store predictions
    if is_dual_output:
        results['y_pred'] = {'sbp': y_pred_sbp, 'dbp': y_pred_dbp}
        results['y_true'] = y_test
        
        # Calculate metrics for both SBP and DBP
        metrics_sbp = calculate_comprehensive_metrics(y_test['sbp'], y_pred_sbp)
        metrics_dbp = calculate_comprehensive_metrics(y_test['dbp'], y_pred_dbp)
        
        results['metrics'] = {
            'sbp': metrics_sbp,
            'dbp': metrics_dbp
        }
        
        # Print metrics for both
        print(f"\n{'='*60}")
        print(f"SBP (Systolic Blood Pressure) Metrics - {dataset_name}")
        print(f"{'='*60}")
        print_metrics(metrics_sbp, f"{dataset_name} SBP")
        
        print(f"\n{'='*60}")
        print(f"DBP (Diastolic Blood Pressure) Metrics - {dataset_name}")
        print(f"{'='*60}")
        print_metrics(metrics_dbp, f"{dataset_name} DBP")
        
        # Plot predictions vs actual for both
        save_path = f"{save_dir}/pred_vs_actual_sbp_{dataset_name.lower()}.png" if save_dir else None
        plot_predictions_vs_actual(y_test['sbp'], y_pred_sbp, f"{dataset_name} SBP", save_path)
        
        save_path = f"{save_dir}/pred_vs_actual_dbp_{dataset_name.lower()}.png" if save_dir else None
        plot_predictions_vs_actual(y_test['dbp'], y_pred_dbp, f"{dataset_name} DBP", save_path)
        
        # Plot error distribution for both
        save_path = f"{save_dir}/error_dist_sbp_{dataset_name.lower()}.png" if save_dir else None
        plot_error_distribution(y_test['sbp'], y_pred_sbp, f"{dataset_name} SBP", save_path)
        
        save_path = f"{save_dir}/error_dist_dbp_{dataset_name.lower()}.png" if save_dir else None
        plot_error_distribution(y_test['dbp'], y_pred_dbp, f"{dataset_name} DBP", save_path)
        
    else:
        # Single output (backward compatibility)
        results['y_pred'] = y_pred_sbp
        results['y_true'] = y_test
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(y_test, y_pred_sbp)
        results['metrics'] = metrics
        print_metrics(metrics, dataset_name)
        
        # Plot predictions vs actual
        save_path = f"{save_dir}/pred_vs_actual_{dataset_name.lower()}.png" if save_dir else None
        plot_predictions_vs_actual(y_test, y_pred_sbp, dataset_name, save_path)
        
        # Plot error distribution
        save_path = f"{save_dir}/error_dist_{dataset_name.lower()}.png" if save_dir else None
        plot_error_distribution(y_test, y_pred_sbp, dataset_name, save_path)
    
    # Visualize attention if available
    if attention_weights is not None and visualize_attention:
        print("\n🔍 Analyzing PAT-based attention mechanism...")
        save_path = f"{save_dir}/attention_{dataset_name.lower()}.png" if save_dir else None
        visualize_attention_weights(attention_weights, num_samples=5, save_path=save_path)
        
        # Analyze attention statistics
        attention_stats = analyze_attention_statistics(attention_weights)
        results['attention_stats'] = attention_stats
    
    return results
