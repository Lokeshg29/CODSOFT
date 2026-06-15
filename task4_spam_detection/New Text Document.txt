# Task 4: Spam SMS Detection

## Overview
Built a machine learning model to classify SMS messages as spam or legitimate (ham) using TF-IDF vectorization and Naive Bayes classification.

## Dataset
SMS Spam Collection Dataset

## Approach
1. Cleaned text data (lowercase, removed punctuation and numbers)
2. Converted text to numerical features using TF-IDF
3. Split data into 80% training and 20% testing
4. Trained a Multinomial Naive Bayes classifier
5. Evaluated using accuracy, precision, recall, and confusion matrix

## Results
- Model: Multinomial Naive Bayes
- Accuracy: 97.31%
- Confusion matrix saved as `confusion_matrix.png`

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn