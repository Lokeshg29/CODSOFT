# Task 1: Movie Genre Classification

## Overview
Built a machine learning model to predict a movie's genre based on its title and plot summary using TF-IDF vectorization (with bigrams) and Logistic Regression.

## Dataset
Genre Classification Dataset (IMDb)

## Approach
1. Combined movie title + description for richer text signal
2. Cleaned text (lowercase, removed punctuation/numbers)
3. TF-IDF vectorization with unigrams + bigrams (20,000 features)
4. Trained Naive Bayes, Logistic Regression, and SVM classifiers
5. Selected Logistic Regression as the best model
6. Evaluated using accuracy, classification report, and confusion matrix

## Results
- Best Model: Logistic Regression
- Validation Accuracy: 59%
- Note: Performance varies significantly by genre due to class imbalance — common genres (documentary, drama, western) score well (F1 0.6-0.8), while rare genres (war, history, biography) have very few training samples and score near 0.
- Confusion matrix saved as `confusion_matrix.png`

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn