# Movie Recommendation System

CS5100 Final Project - Dongni Zeng

## Project Overview
Implementing and evaluating recommendation algorithms using MovieLens dataset:
1. Popularity-based recommendation (baseline)
2. Collaborative filtering (user-based and item-based)
3. Content-based recommendation

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
Download MovieLens 100K from: https://grouplens.org/datasets/movielens/100k/
Place in `data/raw/ml-100k/`

### 3. Run Notebooks
Start with `notebooks/01_data_exploration.ipynb`

## Project Structure
- `data/` - Dataset storage
- `notebooks/` - Jupyter notebooks for exploration
- `src/` - Python source code
- `results/` - Output figures and metrics
- `docs/` - Documentation and progress reports
```

4. SAVE **Cmd+S**）

---

## 📝 **Step 3: create .gitignore**(optional but recommend)

1. New File → `.gitignore`
2. Paste：
```
# Data files (too large for git)
data/raw/
data/processed/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv/
env/

# Jupyter
.ipynb_checkpoints/

# macOS
.DS_Store

# Results (regeneratable)
results/figures/*.png
results/metrics/*.csv