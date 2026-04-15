"""
Advanced Evaluation Metrics
Precision@K, Recall@K, Coverage, and Cold-start Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def precision_at_k(predictions, actual_relevant, k=10):
    """
    Calculate Precision@K
    
    Parameters:
    -----------
    predictions : list
        List of predicted item IDs (ranked)
    actual_relevant : set
        Set of actually relevant item IDs
    k : int
        Number of top predictions to consider
        
    Returns:
    --------
    float
        Precision@K score
    """
    if len(predictions) == 0:
        return 0.0
    
    top_k = predictions[:k]
    relevant_in_top_k = len(set(top_k) & actual_relevant)
    
    return relevant_in_top_k / k


def recall_at_k(predictions, actual_relevant, k=10):
    """
    Calculate Recall@K
    
    Parameters:
    -----------
    predictions : list
        List of predicted item IDs (ranked)
    actual_relevant : set
        Set of actually relevant item IDs
    k : int
        Number of top predictions to consider
        
    Returns:
    --------
    float
        Recall@K score
    """
    if len(actual_relevant) == 0:
        return 0.0
    
    top_k = predictions[:k]
    relevant_in_top_k = len(set(top_k) & actual_relevant)
    
    return relevant_in_top_k / len(actual_relevant)


def average_precision_at_k(predictions, actual_relevant, k=10):
    """
    Calculate Average Precision@K (AP@K)
    
    Parameters:
    -----------
    predictions : list
        List of predicted item IDs (ranked)
    actual_relevant : set
        Set of actually relevant item IDs
    k : int
        Number of top predictions to consider
        
    Returns:
    --------
    float
        Average Precision@K score
    """
    if len(actual_relevant) == 0:
        return 0.0
    
    top_k = predictions[:k]
    
    precisions = []
    num_hits = 0
    
    for i, item in enumerate(top_k, 1):
        if item in actual_relevant:
            num_hits += 1
            precisions.append(num_hits / i)
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / min(len(actual_relevant), k)


def evaluate_ranking_metrics(model, test_data, train_data, k_values=[5, 10, 20], 
                             relevance_threshold=4.0):
    """
    Evaluate ranking metrics for a recommendation model
    
    Parameters:
    -----------
    model : object
        Trained recommendation model with recommend() method
    test_data : pd.DataFrame
        Test data
    train_data : pd.DataFrame
        Training data
    k_values : list
        List of K values to evaluate
    relevance_threshold : float
        Minimum rating to consider an item relevant
        
    Returns:
    --------
    dict
        Dictionary with metrics for each K
    """
    print(f"Evaluating ranking metrics (relevance threshold: {relevance_threshold})...")
    
    # Get unique users in test set
    test_users = test_data['user_id'].unique()
    
    results = {k: {'precision': [], 'recall': [], 'ap': []} for k in k_values}
    
    for user_id in test_users[:100]:  # Sample 100 users for efficiency
        # Get relevant items (test items with rating >= threshold)
        user_test = test_data[test_data['user_id'] == user_id]
        relevant_items = set(user_test[user_test['rating'] >= relevance_threshold]['item_id'])
        
        if len(relevant_items) == 0:
            continue
        
        # Get already rated items (to exclude from recommendations)
        user_train = train_data[train_data['user_id'] == user_id]
        rated_items = set(user_train['item_id'])
        
        # Get recommendations
        try:
            recommendations = model.recommend(user_id, n=max(k_values))
            pred_items = [item_id for item_id, _ in recommendations if item_id not in rated_items]
        except:
            continue
        
        # Calculate metrics for each K
        for k in k_values:
            precision = precision_at_k(pred_items, relevant_items, k)
            recall = recall_at_k(pred_items, relevant_items, k)
            ap = average_precision_at_k(pred_items, relevant_items, k)
            
            results[k]['precision'].append(precision)
            results[k]['recall'].append(recall)
            results[k]['ap'].append(ap)
    
    # Average results
    summary = {}
    for k in k_values:
        summary[f'Precision@{k}'] = np.mean(results[k]['precision']) if results[k]['precision'] else 0
        summary[f'Recall@{k}'] = np.mean(results[k]['recall']) if results[k]['recall'] else 0
        summary[f'AP@{k}'] = np.mean(results[k]['ap']) if results[k]['ap'] else 0
    
    return summary


def catalog_coverage(recommendations_dict, total_items):
    """
    Calculate catalog coverage
    
    Parameters:
    -----------
    recommendations_dict : dict
        Dictionary mapping user_id to list of recommended items
    total_items : int
        Total number of items in catalog
        
    Returns:
    --------
    float
        Coverage percentage
    """
    all_recommended = set()
    for items in recommendations_dict.values():
        all_recommended.update(items)
    
    return len(all_recommended) / total_items


def analyze_cold_start(model, train_data, test_data, rating_thresholds=[5, 10, 20]):
    """
    Analyze model performance on cold-start users
    
    Parameters:
    -----------
    model : object
        Trained model
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    rating_thresholds : list
        Thresholds for categorizing users by number of ratings
        
    Returns:
    --------
    dict
        Performance by user activity level
    """
    print("Analyzing cold-start performance...")
    
    # Count ratings per user in training set
    user_rating_counts = train_data.groupby('user_id').size()
    
    results = {}
    
    for threshold in rating_thresholds:
        # Get users with <= threshold ratings
        cold_users = user_rating_counts[user_rating_counts <= threshold].index
        
        # Get test data for these users
        cold_test = test_data[test_data['user_id'].isin(cold_users)]
        
        if len(cold_test) == 0:
            continue
        
        # Evaluate on cold users
        metrics = model.evaluate(cold_test.sample(min(1000, len(cold_test)), random_state=42))
        
        results[f'<={threshold} ratings'] = {
            'n_users': len(cold_users),
            'n_test_samples': len(cold_test),
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        }
    
    # Also evaluate on warm users (> max threshold)
    warm_users = user_rating_counts[user_rating_counts > max(rating_thresholds)].index
    warm_test = test_data[test_data['user_id'].isin(warm_users)]
    
    if len(warm_test) > 0:
        metrics = model.evaluate(warm_test.sample(min(1000, len(warm_test)), random_state=42))
        results[f'>{max(rating_thresholds)} ratings'] = {
            'n_users': len(warm_users),
            'n_test_samples': len(warm_test),
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        }
    
    return results


def plot_cold_start_analysis(cold_start_results, save_path='results/figures/cold_start_analysis.png'):
    """
    Plot cold-start analysis results
    
    Parameters:
    -----------
    cold_start_results : dict
        Results from analyze_cold_start()
    save_path : str
        Path to save figure
    """
    categories = list(cold_start_results.keys())
    rmse_values = [cold_start_results[cat]['RMSE'] for cat in categories]
    mae_values = [cold_start_results[cat]['MAE'] for cat in categories]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE plot
    ax1 = axes[0]
    bars1 = ax1.bar(categories, rmse_values, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_xlabel('User Activity Level', fontsize=12, fontweight='bold')
    ax1.set_title('Cold-Start Analysis - RMSE', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # MAE plot
    ax2 = axes[1]
    bars2 = ax2.bar(categories, mae_values, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax2.set_xlabel('User Activity Level', fontsize=12, fontweight='bold')
    ax2.set_title('Cold-Start Analysis - MAE', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()


def analyze_parameter_sensitivity(train_data, test_data, k_values=[5, 10, 20, 30, 40, 50]):
    """
    Analyze how K parameter affects Item-based CF performance
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    k_values : list
        List of K values to test
        
    Returns:
    --------
    pd.DataFrame
        Results for different K values
    """
    from collaborative_filtering import ItemBasedCF
    
    print("Analyzing parameter sensitivity (K values)...")
    
    results = []
    test_sample = test_data.sample(min(5000, len(test_data)), random_state=42)
    
    for k in k_values:
        print(f"\nTesting K={k}...")
        model = ItemBasedCF(k=k)
        model.fit(train_data)
        metrics = model.evaluate(test_sample)
        
        results.append({
            'K': k,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        })
    
    return pd.DataFrame(results)


def plot_parameter_sensitivity(results_df, save_path='results/figures/parameter_sensitivity.png'):
    """
    Plot parameter sensitivity analysis
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from analyze_parameter_sensitivity()
    save_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE plot
    ax1 = axes[0]
    ax1.plot(results_df['K'], results_df['RMSE'], marker='o', linewidth=2, 
            markersize=8, color='steelblue')
    ax1.set_xlabel('K (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('Parameter Sensitivity - RMSE', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # MAE plot
    ax2 = axes[1]
    ax2.plot(results_df['K'], results_df['MAE'], marker='s', linewidth=2,
            markersize=8, color='coral')
    ax2.set_xlabel('K (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax2.set_title('Parameter Sensitivity - MAE', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()


# Main analysis script
if __name__ == "__main__":
    print("Advanced Evaluation Analysis\n")
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # Train Item-based CF (best model)
    from collaborative_filtering import ItemBasedCF
    
    print("\nTraining Item-based CF...")
    model = ItemBasedCF(k=20)
    model.fit(train_data)
    
    # Cold-start analysis
    print("\n" + "="*60)
    print("COLD-START ANALYSIS")
    print("="*60)
    cold_start_results = analyze_cold_start(model, train_data, test_data)
    
    for category, metrics in cold_start_results.items():
        print(f"\n{category}:")
        print(f"  Users: {metrics['n_users']}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
    
    plot_cold_start_analysis(cold_start_results)
    
    # Parameter sensitivity
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    sensitivity_results = analyze_parameter_sensitivity(train_data, test_data)
    print("\nResults:")
    print(sensitivity_results.to_string(index=False))
    
    sensitivity_results.to_csv('results/metrics/parameter_sensitivity.csv', index=False)
    plot_parameter_sensitivity(sensitivity_results)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)