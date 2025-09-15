# 🛍️ Product Recommendation System using KNN

This is a simple **product recommender system** built using the **K-Nearest Neighbors (KNN)** algorithm.

## 🚀 Features

- User-based collaborative filtering
- Cosine similarity for measuring user closeness
- Recommends unrated products that similar users liked

## 🧠 How it Works

1. Load dummy ratings from 4 users.
2. Use KNN to find users with similar tastes.
3. Recommend products the user hasn't rated, but their neighbors liked.

## 🛠️ Requirements

- Python 3
- pandas
- scikit-learn

## ▶️ Run the Code

```bash
python main.py