# Movie Recommendation System

CS5100 Final Project - Dongni Zeng

## Overview
Implementation and evaluation of recommendation algorithms using MovieLens 100K dataset.

**Algorithms Implemented:**
- Baseline models (Global Mean, User Mean, Popularity-based)
- Item-based Collaborative Filtering
- User-based Collaborative Filtering  
- Matrix Factorization (SVD)

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** This project requires `numpy<2.0` for compatibility with scikit-surprise.

### 2. Download Data
1. Download MovieLens 100K from https://grouplens.org/datasets/movielens/100k/
2. Extract and place the `ml-100k` folder in `data/raw/`

### 3. Run Code
```bash
# Data exploration
jupyter notebook notebooks/01_data_exploration.ipynb

# Test baseline models
python3 src/baseline.py

# Test collaborative filtering
python3 src/collaborative_filtering.py

# Test matrix factorization
python3 src/matrix_factorization.py

# Evaluate all models
python3 src/evaluate_models.py
```

## Results

| Model | RMSE | MAE |
|-------|------|-----|
| SVD | 0.9256 | 0.7259 |
| Item-based CF | 0.9196 | 0.7154 |
| User-based CF | 0.9988 | 0.7928 |
| Popularity | 1.0210 | 0.8123 |
| User Mean | 1.0417 | 0.8346 |
| Global Mean | 1.1239 | 0.9420 |

**Best Model:** Item-based CF achieves 18.1% improvement over baseline.

## Requirements
- Python 3.8+
- pandas, numpy (<2.0), scikit-learn
- scikit-surprise
- matplotlib, seaborn
- jupyter

See `requirements.txt` for complete list.

## License
Academic project for CS5100 - Foundations of Artificial Intelligence