# Credit Card Fraud Detection

A machine learning project focused on identifying fraudulent transactions in credit card data using Python, pandas and Scikit-learn.

---

## Overview

This project tackles the challenge of **credit card fraud detection** using a real-world, anonymized dataset. Fraudulent transactions are rare, making this a classic case of **imbalanced classification**. The goal is to accurately identify fraud while minimizing false positives and negatives.


## Technologies Used

* **Python 3.12.7**
* **Pandas** & **NumPy** – Data manipulation and analysis
* **Scikit-learn** – Machine Learning (Logistic Regression, train-test split, evaluation metrics)
* **Jupyter Notebook** – Interactive code development


## Dataset

* Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* Features: 30 (anonymized), including `Amount` and `Class` (0 = legit, 1 = fraud)
* Class Imbalance: Only \~0.17% transactions are fraudulent


## Key Steps

1. **Data Loading & Exploration**

   * Checked dataset structure, missing values, and class distribution
   * Used `.info()`, `.describe()`, and `value_counts()` for profiling

2. **Data Balancing**

   * Applied **downsampling** to balance legit and fraud samples (492 each)
   * Ensured equal class representation for training

3. **Model Building**

   * Trained a **Logistic Regression** model
   * Used `train_test_split()` to separate training and testing data
   * Evaluated model performance with **accuracy**, **precision**, and **recall**

---

## Results

* Successfully trained a logistic regression model capable of detecting fraud in a balanced dataset
* Achieved high accuracy with strong recall, prioritizing detection of fraudulent cases
* Built a modular, interpretable, and efficient machine learning pipeline
