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

# Advanced analysis (cold-start and parameter sensitivity)
python3 src/advanced_evaluation.py
```

## Results

| Model | RMSE | MAE |
|-------|------|-----|
| Item-based CF | 0.9254 | 0.7227 |
| SVD | 0.9270 | 0.7306 |
| User-based CF | 1.0067 | 0.8010 |
| Popularity | 1.0158 | 0.8106 |
| User Mean | 1.0394 | 0.8317 |
| Global Mean | 1.1202 | 0.9399 |

**Best Model:** Item-based CF achieves 17.39% improvement over baseline.

## Project Structure
```
recommendation_system/
├── data/
│   ├── raw/ml-100k/        # MovieLens dataset
│   └── processed/          # Train/test splits
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── baseline.py         # Baseline models
│   ├── collaborative_filtering.py  # CF algorithms
│   ├── matrix_factorization.py     # SVD implementation
│   ├── evaluate_models.py  # Model evaluation
│   └── advanced_evaluation.py      # Advanced analysis
├── notebooks/
│   └── 01_data_exploration.ipynb   # Data analysis
├── results/
│   ├── figures/            # Visualization plots
│   └── metrics/            # Performance metrics
└── requirements.txt
```

## Requirements
- Python 3.8+
- pandas, numpy (<2.0), scikit-learn
- scikit-surprise
- matplotlib, seaborn
- jupyter

See `requirements.txt` for complete list.

## Reference
Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. Proceedings of the 10th International Conference on World Wide Web, 285-295.