"""
Comprehensive Model Evaluation and Comparison
Compares all implemented recommendation algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Import all models
from baseline import GlobalMeanBaseline, UserMeanBaseline, PopularityRecommender
from collaborative_filtering import ItemBasedCF, UserBasedCF
from matrix_factorization import MatrixFactorizationSVD


def evaluate_all_models(train_data, test_data, save_results=True):
    """
    Evaluate all recommendation models
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    save_results : bool
        Whether to save results to CSV
        
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    # Use sample for faster evaluation (adjust as needed)
    test_sample = test_data.sample(n=min(10000, len(test_data)), random_state=42)
    
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    print(f"Train set: {len(train_data):,} ratings")
    print(f"Test set: {len(test_sample):,} ratings")
    print("="*70 + "\n")
    
    results = []
    
    # ========== BASELINE MODELS ==========
    print("BASELINE MODELS")
    print("-"*70)
    
    # Global Mean
    print("\n1. Global Mean Baseline")
    global_mean = GlobalMeanBaseline()
    global_mean.fit(train_data)
    metrics = global_mean.evaluate(test_sample)
    results.append({
        'Category': 'Baseline',
        'Model': 'Global Mean',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"   RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
    
    # User Mean
    print("\n2. User Mean Baseline")
    user_mean = UserMeanBaseline()
    user_mean.fit(train_data)
    metrics = user_mean.evaluate(test_sample)
    results.append({
        'Category': 'Baseline',
        'Model': 'User Mean',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"   RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
    
    # Popularity
    print("\n3. Popularity-based")
    popularity = PopularityRecommender(min_ratings=5)
    popularity.fit(train_data)
    metrics = popularity.evaluate(test_sample)
    results.append({
        'Category': 'Baseline',
        'Model': 'Popularity',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"   RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
    
    # ========== COLLABORATIVE FILTERING ==========
    print("\n\n" + "="*70)
    print("COLLABORATIVE FILTERING MODELS")
    print("-"*70)
    
    # Item-based CF
    print("\n4. Item-based CF (k=20)")
    item_cf = ItemBasedCF(k=20)
    item_cf.fit(train_data)
    metrics = item_cf.evaluate(test_sample)
    results.append({
        'Category': 'Collaborative Filtering',
        'Model': 'Item-based CF',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"   RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
    
    # User-based CF
    print("\n5. User-based CF (k=20)")
    user_cf = UserBasedCF(k=20)
    user_cf.fit(train_data)
    metrics = user_cf.evaluate(test_sample)
    results.append({
        'Category': 'Collaborative Filtering',
        'Model': 'User-based CF',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"   RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
    
    # SVD
    print("\n6. Matrix Factorization (SVD)")
    svd = MatrixFactorizationSVD(n_factors=50, n_epochs=20)
    svd.fit(train_data)
    metrics = svd.evaluate(test_sample)
    results.append({
        'Category': 'Matrix Factorization',
        'Model': 'SVD',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"   RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
    
    # ========== RESULTS SUMMARY ==========
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    
    # Sort by RMSE
    results_df = results_df.sort_values('RMSE')
    
    print(results_df.to_string(index=False))
    print("="*70)
    
    # Find best model
    best_model = results_df.iloc[0]
    print(f"\nBest Model: {best_model['Model']}")
    print(f"  RMSE: {best_model['RMSE']:.4f}")
    print(f"  MAE:  {best_model['MAE']:.4f}")
    
    # Calculate improvement over baseline
    baseline_rmse = results_df[results_df['Model'] == 'Global Mean']['RMSE'].values[0]
    best_rmse = best_model['RMSE']
    improvement = (baseline_rmse - best_rmse) / baseline_rmse * 100
    print(f"\nImprovement over Global Mean Baseline: {improvement:.2f}%")
    
    # Save results
    if save_results:
        results_df.to_csv('results/metrics/all_models_comparison.csv', index=False)
        print("\nResults saved to results/metrics/all_models_comparison.csv")
    
    return results_df


def plot_model_comparison(results_df, save_fig=True):
    """
    Create visualization comparing all models
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from evaluate_all_models
    save_fig : bool
        Whether to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    ax1 = axes[0]
    colors = ['#ff9999' if cat == 'Baseline' else '#66b3ff' if cat == 'Collaborative Filtering' else '#99ff99' 
              for cat in results_df['Category']]
    
    bars1 = ax1.barh(results_df['Model'], results_df['RMSE'], color=colors, edgecolor='black')
    ax1.set_xlabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, results_df['RMSE'])):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10)
    
    # MAE comparison
    ax2 = axes[1]
    bars2 = ax2.barh(results_df['Model'], results_df['MAE'], color=colors, edgecolor='black')
    ax2.set_xlabel('MAE (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Comparison - MAE', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, results_df['MAE'])):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff9999', edgecolor='black', label='Baseline'),
        Patch(facecolor='#66b3ff', edgecolor='black', label='Collaborative Filtering'),
        Patch(facecolor='#99ff99', edgecolor='black', label='Matrix Factorization')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
              bbox_to_anchor=(0.5, 0.98), fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_fig:
        plt.savefig('results/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        print("Figure saved to results/figures/model_comparison.png")
    
    plt.show()


def main():
    """Main evaluation function"""
    print("Loading data...")
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # Evaluate all models
    results_df = evaluate_all_models(train_data, test_data, save_results=True)
    
    # Create visualization
    print("\nGenerating comparison plot...")
    plot_model_comparison(results_df, save_fig=True)
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
