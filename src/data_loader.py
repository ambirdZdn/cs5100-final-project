"""
数据加载和预处理模块
MovieLens 100K数据集工具函数
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class MovieLensLoader:
    """MovieLens 100K数据集加载器"""
    
    def __init__(self, data_path='data/raw/ml-100k'):
        """
        初始化数据加载器
        
        Parameters:
        -----------
        data_path : str
            MovieLens数据集路径
        """
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.users = None
        
    def load_ratings(self):
        """
        加载评分数据
        
        Returns:
        --------
        pd.DataFrame
            包含user_id, item_id, rating, timestamp的DataFrame
        """
        print("📊 Loading ratings data...")
        
        ratings_file = self.data_path / 'u.data'
        
        self.ratings = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        print(f"✅ Loaded {len(self.ratings):,} ratings")
        return self.ratings
    
    def load_movies(self):
        """
        加载电影信息
        
        Returns:
        --------
        pd.DataFrame
            包含电影ID、标题、类型等信息的DataFrame
        """
        print("🎬 Loading movies data...")
        
        movies_file = self.data_path / 'u.item'
        
        # 定义列名
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
        
        print(f"✅ Loaded {len(self.movies):,} movies")
        return self.movies
    
    def load_users(self):
        """
        加载用户信息
        
        Returns:
        --------
        pd.DataFrame
            包含用户ID、年龄、性别、职业等信息的DataFrame
        """
        print("👥 Loading users data...")
        
        users_file = self.data_path / 'u.user'
        
        self.users = pd.read_csv(
            users_file,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            engine='python'
        )
        
        print(f"✅ Loaded {len(self.users):,} users")
        return self.users
    
    def load_all(self):
        """
        加载所有数据
        
        Returns:
        --------
        tuple
            (ratings, movies, users)的元组
        """
        self.load_ratings()
        self.load_movies()
        self.load_users()
        
        return self.ratings, self.movies, self.users
    
    def get_data_stats(self):
        """
        获取数据集统计信息
        
        Returns:
        --------
        dict
            包含各种统计信息的字典
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
        """打印数据集统计信息"""
        stats = self.get_data_stats()
        
        print("\n" + "="*50)
        print("📊 MovieLens 100K Dataset Statistics")
        print("="*50)
        print(f"Total Ratings:     {stats['n_ratings']:,}")
        print(f"Number of Users:   {stats['n_users']:,}")
        print(f"Number of Movies:  {stats['n_movies']:,}")
        print(f"\nRating Range:      {stats['rating_min']} - {stats['rating_max']}")
        print(f"Average Rating:    {stats['rating_mean']:.2f} ± {stats['rating_std']:.2f}")
        print(f"Data Sparsity:     {stats['sparsity']:.2%}")
        print("="*50 + "\n")
        
        # 额外统计
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
        创建用户-电影评分矩阵
        
        Returns:
        --------
        pd.DataFrame
            用户为行，电影为列的评分矩阵（稀疏）
        """
        if self.ratings is None:
            self.load_ratings()
        
        print("🔨 Creating rating matrix...")
        
        rating_matrix = self.ratings.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        )
        
        print(f"✅ Matrix shape: {rating_matrix.shape}")
        print(f"   ({rating_matrix.shape[0]} users × {rating_matrix.shape[1]} movies)")
        
        return rating_matrix
    
    def split_train_test(self, test_size=0.2, random_state=42):
        """
        划分训练集和测试集
        
        Parameters:
        -----------
        test_size : float
            测试集比例 (default: 0.2)
        random_state : int
            随机种子 (default: 42)
        
        Returns:
        --------
        tuple
            (train_data, test_data)
        """
        if self.ratings is None:
            self.load_ratings()
        
        print(f"✂️  Splitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
        
        train_data, test_data = train_test_split(
            self.ratings,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"✅ Train set: {len(train_data):,} ratings")
        print(f"✅ Test set:  {len(test_data):,} ratings")
        
        return train_data, test_data
    
    def get_genre_matrix(self):
        """
        获取电影类型矩阵（用于content-based推荐）
        
        Returns:
        --------
        pd.DataFrame
            电影ID为索引，19个类型为列的二进制矩阵
        """
        if self.movies is None:
            self.load_movies()
        
        # 类型列名
        genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        genre_matrix = self.movies.set_index('item_id')[genre_columns]
        
        return genre_matrix


def quick_load(data_path='data/raw/ml-100k'):
    """
    快速加载所有数据的便捷函数
    
    Parameters:
    -----------
    data_path : str
        数据路径
    
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


# 测试代码
if __name__ == "__main__":
    # 测试数据加载
    print("Testing MovieLensLoader...")
    
    loader = MovieLensLoader()
    
    # 加载所有数据
    ratings, movies, users = loader.load_all()
    
    # 打印统计信息
    loader.print_stats()
    
    # 测试数据划分
    train, test = loader.split_train_test()
    
    print("✅ All tests passed!")