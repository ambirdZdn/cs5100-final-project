"""
Data Loading and Preprocessing Module
MovieLens 100K Dataset Utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class MovieLensLoader:
    """MovieLens 100K Dataset Loader"""
    
    def __init__(self, data_path='data/raw/ml-100k'):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_path : str
            Path to MovieLens dataset
        """
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.users = None
        
    def load_ratings(self):
        """
        Load ratings data
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing user_id, item_id, rating, timestamp
        """
        print("Loading ratings data...")
        
        ratings_file = self.data_path / 'u.data'
        
        self.ratings = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        print(f"Loaded {len(self.ratings):,} ratings")
        return self.ratings
    
    def load_movies(self):
        """
        Load movie information
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing movie ID, title, genres, etc.
        """
        print("Loading movies data...")
        
        movies_file = self.data_path / 'u.item'
        
        # Define column names
        columns = [
            'item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        self.movies = pd.read_csv(
            movies_file,
            sep='|',
            encoding='latin-1',
            names=columns,
            engine='python'
        )
        
        print(f"Loaded {len(self.movies):,} movies")
        return self.movies
    
    def load_users(self):
        """
        Load user information
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing user ID, age, gender, occupation, etc.
        """
        print("Loading users data...")
        
        users_file = self.data_path / 'u.user'
        
        self.users = pd.read_csv(
            users_file,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            engine='python'
        )
        
        print(f"Loaded {len(self.users):,} users")
        return self.users
    
    def load_all(self):
        """
        Load all data
        
        Returns:
        --------
        tuple
            Tuple of (ratings, movies, users)
        """
        self.load_ratings()
        self.load_movies()
        self.load_users()
        
        return self.ratings, self.movies, self.users
    
    def get_data_stats(self):
        """
        Get dataset statistics
        
        Returns:
        --------
        dict
            Dictionary containing various statistics
        """
        if self.ratings is None:
            self.load_ratings()
        
        stats = {
            'n_ratings': len(self.ratings),
            'n_users': self.ratings['user_id'].nunique(),
            'n_movies': self.ratings['item_id'].nunique(),
            'rating_min': self.ratings['rating'].min(),
            'rating_max': self.ratings['rating'].max(),
            'rating_mean': self.ratings['rating'].mean(),
            'rating_std': self.ratings['rating'].std(),
            'sparsity': 1 - (len(self.ratings) / 
                           (self.ratings['user_id'].nunique() * 
                            self.ratings['item_id'].nunique()))
        }
        
        return stats
    
    def print_stats(self):
        """Print dataset statistics"""
        stats = self.get_data_stats()
        
        print("\n" + "="*50)
        print("MovieLens 100K Dataset Statistics")
        print("="*50)
        print(f"Total Ratings:     {stats['n_ratings']:,}")
        print(f"Number of Users:   {stats['n_users']:,}")
        print(f"Number of Movies:  {stats['n_movies']:,}")
        print(f"\nRating Range:      {stats['rating_min']} - {stats['rating_max']}")
        print(f"Average Rating:    {stats['rating_mean']:.2f} ± {stats['rating_std']:.2f}")
        print(f"Data Sparsity:     {stats['sparsity']:.2%}")
        print("="*50 + "\n")
        
        # Additional statistics
        if self.ratings is not None:
            ratings_per_user = self.ratings.groupby('user_id').size()
            ratings_per_movie = self.ratings.groupby('item_id').size()
            
            print("User Activity:")
            print(f"  Avg ratings/user:  {ratings_per_user.mean():.1f}")
            print(f"  Min ratings/user:  {ratings_per_user.min()}")
            print(f"  Max ratings/user:  {ratings_per_user.max()}")
            
            print("\nMovie Popularity:")
            print(f"  Avg ratings/movie: {ratings_per_movie.mean():.1f}")
            print(f"  Min ratings/movie: {ratings_per_movie.min()}")
            print(f"  Max ratings/movie: {ratings_per_movie.max()}")
            print()
    
    def create_rating_matrix(self):
        """
        Create user-item rating matrix
        
        Returns:
        --------
        pd.DataFrame
            Rating matrix with users as rows and movies as columns (sparse)
        """
        if self.ratings is None:
            self.load_ratings()
        
        print("Creating rating matrix...")
        
        rating_matrix = self.ratings.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        )
        
        print(f"Matrix shape: {rating_matrix.shape}")
        print(f"({rating_matrix.shape[0]} users × {rating_matrix.shape[1]} movies)")
        
        return rating_matrix
    
    def split_train_test(self, test_size=0.2, random_state=42):
        """
        Split data into training and test sets
        
        Parameters:
        -----------
        test_size : float
            Proportion of test set (default: 0.2)
        random_state : int
            Random seed (default: 42)
        
        Returns:
        --------
        tuple
            (train_data, test_data)
        """
        if self.ratings is None:
            self.load_ratings()
        
        print(f"Splitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
        
        train_data, test_data = train_test_split(
            self.ratings,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"Train set: {len(train_data):,} ratings")
        print(f"Test set:  {len(test_data):,} ratings")
        
        return train_data, test_data
    
    def get_genre_matrix(self):
        """
        Get movie genre matrix (for content-based recommendation)
        
        Returns:
        --------
        pd.DataFrame
            Binary matrix with movie IDs as index and 19 genres as columns
        """
        if self.movies is None:
            self.load_movies()
        
        # Genre column names
        genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        genre_matrix = self.movies.set_index('item_id')[genre_columns]
        
        return genre_matrix


def quick_load(data_path='data/raw/ml-100k'):
    """
    Convenience function to quickly load all data
    
    Parameters:
    -----------
    data_path : str
        Path to data
    
    Returns:
    --------
    tuple
        (ratings, movies, users) DataFrames
    
    Example:
    --------
    >>> ratings, movies, users = quick_load()
    """
    loader = MovieLensLoader(data_path)
    return loader.load_all()


if __name__ == "__main__":
    # Test data loading
    print("Testing MovieLensLoader...\n")
    
    loader = MovieLensLoader()
    
    # Load all data
    ratings, movies, users = loader.load_all()
    
    # Print statistics
    loader.print_stats()
    
    # Test train/test split
    train, test = loader.split_train_test()
    
    print("\nAll tests passed!")