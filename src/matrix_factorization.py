"""
Matrix Factorization Recommendation Algorithm
Implementation using SVD (Singular Value Decomposition)
Uses the Surprise library for efficient implementation
"""

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise import accuracy
from surprise.model_selection import GridSearchCV
import time


class MatrixFactorizationSVD:
    """
    Matrix Factorization using SVD
    
    Decomposes the user-item rating matrix into user and item latent factor matrices.
    Often provides the best accuracy among collaborative filtering methods.
    
    Uses the Surprise library implementation of SVD.
    """
    
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Initialize SVD model
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors (default: 50)
        n_epochs : int
            Number of training epochs (default: 20)
        lr_all : float
            Learning rate for all parameters (default: 0.005)
        reg_all : float
            Regularization parameter for all parameters (default: 0.02)
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.model = None
        self.trainset = None
        
    def fit(self, train_data):
        """
        Fit the SVD model
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with columns [user_id, item_id, rating]
        """
        print("Training Matrix Factorization (SVD)...")
        print(f"Parameters: n_factors={self.n_factors}, n_epochs={self.n_epochs}")
        start_time = time.time()
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_data[['user_id', 'item_id', 'rating']], 
            reader
        )
        
        # Build full trainset
        self.trainset = data.build_full_trainset()
        
        # Initialize and train SVD model
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42,
            verbose=False
        )
        
        self.model.fit(self.trainset)
        
        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed:.2f} seconds")
        
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
        prediction = self.model.predict(user_id, item_id)
        return prediction.est
    
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
        # Prepare test data for Surprise
        reader = Reader(rating_scale=(1, 5))
        testset = [
            (row['user_id'], row['item_id'], row['rating']) 
            for _, row in test_data.iterrows()
        ]
        
        # Get predictions
        predictions = self.model.test(testset)
        
        # Calculate metrics
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }
    
    def recommend(self, user_id, n=10, items_to_exclude=None):
        """
        Recommend top N items for a user
        
        Parameters:
        -----------
        user_id : int
            User ID
        n : int
            Number of recommendations
        items_to_exclude : list
            List of item IDs to exclude (e.g., already rated items)
            
        Returns:
        --------
        list
            List of (item_id, predicted_rating) tuples
        """
        # Get all items
        all_items = self.trainset.all_items()
        
        # Exclude items if specified
        if items_to_exclude:
            items = [item for item in all_items if item not in items_to_exclude]
        else:
            items = all_items
        
        # Predict ratings for all items
        predictions = [
            (self.trainset.to_raw_iid(item), self.predict(user_id, self.trainset.to_raw_iid(item))) 
            for item in items
        ]
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]


def tune_svd_parameters(train_data, param_grid=None):
    """
    Tune SVD hyperparameters using grid search
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    param_grid : dict
        Parameter grid for grid search
        
    Returns:
    --------
    dict
        Best parameters found
    """
    print("Tuning SVD hyperparameters...")
    print("This may take several minutes...\n")
    
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'n_factors': [30, 50, 100],
            'n_epochs': [20, 30],
            'lr_all': [0.002, 0.005],
            'reg_all': [0.02, 0.05]
        }
    
    # Prepare data
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        train_data[['user_id', 'item_id', 'rating']], 
        reader
    )
    
    # Grid search
    gs = GridSearchCV(
        SVD,
        param_grid,
        measures=['rmse', 'mae'],
        cv=3,
        n_jobs=-1,
        joblib_verbose=2
    )
    
    gs.fit(data)
    
    # Print results
    print("\nBest RMSE score:", gs.best_score['rmse'])
    print("Best parameters:", gs.best_params['rmse'])
    
    return gs.best_params['rmse']


def compare_all_cf_models(train_data, test_data):
    """
    Compare all collaborative filtering models
    
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
    from collaborative_filtering import ItemBasedCF, UserBasedCF
    
    print("="*60)
    print("Comparing All Collaborative Filtering Models")
    print("="*60)
    
    # Use a sample of test data for faster evaluation
    test_sample = test_data.sample(n=min(5000, len(test_data)), random_state=42)
    print(f"\nUsing {len(test_sample)} test samples for evaluation\n")
    
    results = []
    
    # Item-based CF
    print("1. Item-based CF")
    print("-" * 40)
    item_cf = ItemBasedCF(k=20)
    item_cf.fit(train_data)
    metrics = item_cf.evaluate(test_sample)
    results.append({
        'Model': 'Item-based CF',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}\n")
    
    # User-based CF
    print("2. User-based CF")
    print("-" * 40)
    user_cf = UserBasedCF(k=20)
    user_cf.fit(train_data)
    metrics = user_cf.evaluate(test_sample)
    results.append({
        'Model': 'User-based CF',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}\n")
    
    # SVD
    print("3. Matrix Factorization (SVD)")
    print("-" * 40)
    svd = MatrixFactorizationSVD(n_factors=50, n_epochs=20)
    svd.fit(train_data)
    metrics = svd.evaluate(test_sample)
    results.append({
        'Model': 'SVD',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    })
    print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}\n")
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    
    print("="*60)
    print("Comparison Summary")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    return results_df


# Test code
if __name__ == "__main__":
    print("Testing Matrix Factorization (SVD)\n")
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # Use sample for faster testing
    test_sample = test_data.sample(n=5000, random_state=42)
    
    print(f"Train set: {len(train_data):,} ratings")
    print(f"Test sample: {len(test_sample):,} ratings\n")
    
    # Test SVD
    print("="*60)
    print("Matrix Factorization (SVD)")
    print("="*60)
    svd = MatrixFactorizationSVD(n_factors=50, n_epochs=20)
    svd.fit(train_data)
    
    print("\nEvaluating...")
    metrics = svd.evaluate(test_sample)
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    
    # Test recommendation
    print("\nTesting recommendation for user 1...")
    recommendations = svd.recommend(user_id=1, n=5)
    print("Top 5 recommendations:")
    for item_id, rating in recommendations[:5]:
        print(f"  Item {item_id}: predicted rating = {rating:.2f}")
    
    print("\nTesting complete!")
