"""
Collaborative Filtering Recommendation Algorithms
Implementation of user-based and item-based collaborative filtering
Based on Sarwar et al. (2001) - Item-based Collaborative Filtering
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import time


class ItemBasedCF:
    """
    Item-based Collaborative Filtering
    
    Recommends items similar to those the user has already rated.
    Uses adjusted cosine similarity to compute item-item similarities.
    
    Reference:
    Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). 
    Item-based collaborative filtering recommendation algorithms.
    """
    
    def __init__(self, k=20, min_support=5):
        """
        Initialize Item-based CF
        
        Parameters:
        -----------
        k : int
            Number of similar items to consider (default: 20)
        min_support : int
            Minimum number of common users for similarity calculation (default: 5)
        """
        self.k = k
        self.min_support = min_support
        self.item_similarity = None
        self.rating_matrix = None
        self.user_means = None
        self.global_mean = None
        self.train_data = None
        
    def fit(self, train_data):
        """
        Fit the model by computing item-item similarities
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with columns [user_id, item_id, rating]
        """
        print("Training Item-based Collaborative Filtering...")
        start_time = time.time()
        
        self.train_data = train_data
        self.global_mean = train_data['rating'].mean()
        
        # Create user-item rating matrix
        print("Creating rating matrix...")
        self.rating_matrix = train_data.pivot(
            index='user_id',
            columns='item_id', 
            values='rating'
        )
        
        # Compute user mean ratings
        self.user_means = self.rating_matrix.mean(axis=1)
        
        # Adjust ratings by subtracting user means (adjusted cosine similarity)
        print("Computing adjusted ratings...")
        rating_matrix_adjusted = self.rating_matrix.sub(self.user_means, axis=0)
        
        # Fill NaN with 0 for similarity computation
        rating_matrix_filled = rating_matrix_adjusted.fillna(0)
        
        # Compute item-item similarity matrix using cosine similarity
        print(f"Computing item similarities (this may take a moment)...")
        self.item_similarity = cosine_similarity(rating_matrix_filled.T)
        
        # Convert to DataFrame for easier access
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.rating_matrix.columns,
            columns=self.rating_matrix.columns
        )
        
        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed:.2f} seconds")
        print(f"Computed similarities for {len(self.item_similarity)} items")
        
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
            Predicted rating
        """
        # Handle cold start - item not in training set
        if item_id not in self.item_similarity.index:
            return self.global_mean
        
        # Handle cold start - user not in training set
        if user_id not in self.rating_matrix.index:
            return self.global_mean
        
        # Get items rated by this user
        user_ratings = self.rating_matrix.loc[user_id]
        rated_items = user_ratings.dropna()
        
        if len(rated_items) == 0:
            return self.global_mean
        
        # Get similarities between target item and rated items
        similarities = self.item_similarity.loc[item_id, rated_items.index]
        
        # Sort by similarity and take top k
        top_similar = similarities.nlargest(self.k)
        
        # Remove items with zero or negative similarity
        top_similar = top_similar[top_similar > 0]
        
        if len(top_similar) == 0:
            return self.global_mean
        
        # Compute weighted average
        numerator = sum(sim * rated_items[item] for item, sim in top_similar.items())
        denominator = sum(abs(sim) for sim in top_similar.values)
        
        if denominator == 0:
            return self.global_mean
        
        prediction = numerator / denominator
        
        # Clip to valid rating range
        prediction = np.clip(prediction, 1, 5)
        
        return prediction
    
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
        print("Generating predictions...")
        predictions = []
        
        for idx, row in test_data.iterrows():
            pred = self.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            
            # Progress indicator
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1}/{len(test_data)} predictions")
        
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
    
    def recommend(self, user_id, n=10):
        """
        Recommend top N items for a user
        
        Parameters:
        -----------
        user_id : int
            User ID
        n : int
            Number of recommendations
            
        Returns:
        --------
        list
            List of (item_id, predicted_rating) tuples
        """
        # Get all items
        all_items = self.item_similarity.index
        
        # Get items user has already rated
        if user_id in self.rating_matrix.index:
            rated_items = self.rating_matrix.loc[user_id].dropna().index
        else:
            rated_items = []
        
        # Get unrated items
        unrated_items = [item for item in all_items if item not in rated_items]
        
        # Predict ratings for unrated items
        predictions = [(item, self.predict(user_id, item)) for item in unrated_items]
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]


class UserBasedCF:
    """
    User-based Collaborative Filtering
    
    Recommends items liked by similar users.
    Uses cosine similarity to compute user-user similarities.
    """
    
    def __init__(self, k=20, min_support=5):
        """
        Initialize User-based CF
        
        Parameters:
        -----------
        k : int
            Number of similar users to consider (default: 20)
        min_support : int
            Minimum number of common items for similarity calculation (default: 5)
        """
        self.k = k
        self.min_support = min_support
        self.user_similarity = None
        self.rating_matrix = None
        self.global_mean = None
        self.train_data = None
        
    def fit(self, train_data):
        """
        Fit the model by computing user-user similarities
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with columns [user_id, item_id, rating]
        """
        print("Training User-based Collaborative Filtering...")
        start_time = time.time()
        
        self.train_data = train_data
        self.global_mean = train_data['rating'].mean()
        
        # Create user-item rating matrix
        print("Creating rating matrix...")
        self.rating_matrix = train_data.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        )
        
        # Fill NaN with 0 for similarity computation
        rating_matrix_filled = self.rating_matrix.fillna(0)
        
        # Compute user-user similarity matrix using cosine similarity
        print(f"Computing user similarities (this may take a moment)...")
        self.user_similarity = cosine_similarity(rating_matrix_filled)
        
        # Convert to DataFrame
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.rating_matrix.index,
            columns=self.rating_matrix.index
        )
        
        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed:.2f} seconds")
        print(f"Computed similarities for {len(self.user_similarity)} users")
        
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
            Predicted rating
        """
        # Handle cold start - user not in training set
        if user_id not in self.user_similarity.index:
            return self.global_mean
        
        # Handle cold start - item not in training set
        if item_id not in self.rating_matrix.columns:
            return self.global_mean
        
        # Get users who rated this item
        item_ratings = self.rating_matrix[item_id]
        users_who_rated = item_ratings.dropna()
        
        if len(users_who_rated) == 0:
            return self.global_mean
        
        # Get similarities between target user and users who rated the item
        similarities = self.user_similarity.loc[user_id, users_who_rated.index]
        
        # Sort by similarity and take top k
        top_similar = similarities.nlargest(self.k)
        
        # Remove users with zero or negative similarity
        top_similar = top_similar[top_similar > 0]
        
        if len(top_similar) == 0:
            return self.global_mean
        
        # Compute weighted average
        numerator = sum(sim * users_who_rated[uid] for uid, sim in top_similar.items())
        denominator = sum(abs(sim) for sim in top_similar.values)
        
        if denominator == 0:
            return self.global_mean
        
        prediction = numerator / denominator
        
        # Clip to valid rating range
        prediction = np.clip(prediction, 1, 5)
        
        return prediction
    
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
        print("Generating predictions...")
        predictions = []
        
        for idx, row in test_data.iterrows():
            pred = self.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            
            # Progress indicator
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1}/{len(test_data)} predictions")
        
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


# Test code
if __name__ == "__main__":
    print("Testing Collaborative Filtering Algorithms\n")
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # Take a sample for faster testing
    test_sample = test_data.sample(n=5000, random_state=42)
    
    print(f"Train set: {len(train_data):,} ratings")
    print(f"Test sample: {len(test_sample):,} ratings\n")
    
    # Test Item-based CF
    print("="*60)
    print("Item-based Collaborative Filtering")
    print("="*60)
    item_cf = ItemBasedCF(k=20)
    item_cf.fit(train_data)
    
    print("\nEvaluating...")
    metrics = item_cf.evaluate(test_sample)
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    
    # Test User-based CF
    print("\n" + "="*60)
    print("User-based Collaborative Filtering")
    print("="*60)
    user_cf = UserBasedCF(k=20)
    user_cf.fit(train_data)
    
    print("\nEvaluating...")
    metrics = user_cf.evaluate(test_sample)
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    
    print("\nTesting complete!")
