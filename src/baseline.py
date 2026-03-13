"""
Baseline Recommendation Algorithms
Simple baseline models for comparison
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class GlobalMeanBaseline:
    """
    Global Mean Baseline Recommender
    Predicts the global average rating for all user-item pairs
    """
    
    def __init__(self):
        self.global_mean = None
        
    def fit(self, train_data):
        """
        Fit the model by computing global mean rating
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with columns [user_id, item_id, rating]
        """
        self.global_mean = train_data['rating'].mean()
        print(f"Global mean rating: {self.global_mean:.3f}")
        
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair
        
        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
            
        Returns:
        --------
        float
            Predicted rating (global mean)
        """
        return self.global_mean
    
    def predict_all(self, test_data):
        """
        Predict ratings for all user-item pairs in test set
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data with columns [user_id, item_id, rating]
            
        Returns:
        --------
        np.array
            Array of predicted ratings
        """
        return np.full(len(test_data), self.global_mean)
    
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
            
        Returns:
        --------
        dict
            Dictionary with RMSE and MAE
        """
        predictions = self.predict_all(test_data)
        actual = test_data['rating'].values
        
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }


class UserMeanBaseline:
    """
    User Mean Baseline Recommender
    Predicts each user's average rating
    """
    
    def __init__(self):
        self.user_means = None
        self.global_mean = None
        
    def fit(self, train_data):
        """
        Fit the model by computing mean rating per user
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        """
        # Compute mean rating for each user
        self.user_means = train_data.groupby('user_id')['rating'].mean()
        
        # Compute global mean as fallback for new users
        self.global_mean = train_data['rating'].mean()
        
        print(f"Computed mean ratings for {len(self.user_means)} users")
        print(f"Global mean (fallback): {self.global_mean:.3f}")
        
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair
        
        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
            
        Returns:
        --------
        float
            Predicted rating (user's mean or global mean)
        """
        if user_id in self.user_means.index:
            return self.user_means[user_id]
        else:
            return self.global_mean
    
    def predict_all(self, test_data):
        """
        Predict ratings for all user-item pairs in test set
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
            
        Returns:
        --------
        np.array
            Array of predicted ratings
        """
        predictions = []
        for _, row in test_data.iterrows():
            pred = self.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
        return np.array(predictions)
    
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
            
        Returns:
        --------
        dict
            Dictionary with RMSE and MAE
        """
        predictions = self.predict_all(test_data)
        actual = test_data['rating'].values
        
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }


class PopularityRecommender:
    """
    Popularity-based Recommender
    Recommends items with highest average ratings
    Considers rating count to avoid bias from items with few ratings
    """
    
    def __init__(self, min_ratings=5):
        """
        Initialize recommender
        
        Parameters:
        -----------
        min_ratings : int
            Minimum number of ratings required for an item
        """
        self.min_ratings = min_ratings
        self.item_stats = None
        self.global_mean = None
        
    def fit(self, train_data):
        """
        Fit the model by computing item statistics
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        """
        # Compute mean rating and count for each item
        self.item_stats = train_data.groupby('item_id').agg({
            'rating': ['mean', 'count']
        })
        self.item_stats.columns = ['mean_rating', 'rating_count']
        
        # Compute global mean
        self.global_mean = train_data['rating'].mean()
        
        # Filter items with sufficient ratings
        popular_items = self.item_stats[
            self.item_stats['rating_count'] >= self.min_ratings
        ]
        
        print(f"Total items: {len(self.item_stats)}")
        print(f"Items with >= {self.min_ratings} ratings: {len(popular_items)}")
        print(f"Global mean rating: {self.global_mean:.3f}")
        
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair
        
        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
            
        Returns:
        --------
        float
            Predicted rating (item's mean or global mean)
        """
        if item_id in self.item_stats.index:
            return self.item_stats.loc[item_id, 'mean_rating']
        else:
            return self.global_mean
    
    def predict_all(self, test_data):
        """
        Predict ratings for all user-item pairs in test set
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
            
        Returns:
        --------
        np.array
            Array of predicted ratings
        """
        predictions = []
        for _, row in test_data.iterrows():
            pred = self.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
        return np.array(predictions)
    
    def recommend(self, n=10, min_ratings=None):
        """
        Get top N most popular items
        
        Parameters:
        -----------
        n : int
            Number of recommendations
        min_ratings : int
            Minimum rating count (overrides init value if provided)
            
        Returns:
        --------
        pd.DataFrame
            Top N items with their statistics
        """
        min_rat = min_ratings if min_ratings is not None else self.min_ratings
        
        # Filter and sort
        popular = self.item_stats[
            self.item_stats['rating_count'] >= min_rat
        ].sort_values('mean_rating', ascending=False).head(n)
        
        return popular
    
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
            
        Returns:
        --------
        dict
            Dictionary with RMSE and MAE
        """
        predictions = self.predict_all(test_data)
        actual = test_data['rating'].values
        
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }


def compare_baselines(train_data, test_data):
    """
    Compare all baseline models
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
        
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    print("="*60)
    print("Comparing Baseline Models")
    print("="*60)
    
    results = []
    
    # Global Mean Baseline
    print("\n1. Global Mean Baseline")
    print("-" * 40)
    global_mean = GlobalMeanBaseline()
    global_mean.fit(train_data)
    metrics = global_mean.evaluate(test_data)
    results.append({
        'Model': 'Global Mean',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    
    # User Mean Baseline
    print("\n2. User Mean Baseline")
    print("-" * 40)
    user_mean = UserMeanBaseline()
    user_mean.fit(train_data)
    metrics = user_mean.evaluate(test_data)
    results.append({
        'Model': 'User Mean',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    
    # Popularity-based
    print("\n3. Popularity-based Recommender")
    print("-" * 40)
    popularity = PopularityRecommender(min_ratings=5)
    popularity.fit(train_data)
    metrics = popularity.evaluate(test_data)
    results.append({
        'Model': 'Popularity',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("Baseline Comparison Summary")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    return results_df


# Test code
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    print(f"Train set: {len(train_data):,} ratings")
    print(f"Test set:  {len(test_data):,} ratings\n")
    
    # Compare all baselines
    results = compare_baselines(train_data, test_data)
    
    # Save results
    results.to_csv('results/metrics/baseline_results.csv', index=False)
    print("\nResults saved to results/metrics/baseline_results.csv")
