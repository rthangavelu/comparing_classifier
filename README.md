# Bank Marketing Classifier Comparison
## Overview

This project compares the performance of four machine learning classifiers on the Bank Marketing dataset from the UCI Machine Learning Repository.
The goal is to predict whether a bank client will subscribe to a term deposit based on demographic, financial, and campaign-related features.

### Classifiers compared:

K Nearest Neighbors (KNN)
Logistic Regression
Decision Tree
Support Vector Machine (SVM)

We establish a baseline performance and then improve the models using scaling, class balancing, and hyperparameter adjustments.

### Dataset

Source: UCI Bank Marketing Dataset
Period: 17 marketing campaigns (May 2008 – November 2010)
Target Variable:
y: has the client subscribed a term deposit? (yes/no)

### Steps
#### Problem Understanding
  Business objective: identify potential subscribers to improve telemarketing efficiency.
#### Data Preparation
  Load dataset (bank-additional-full.csv).
  Select relevant features (age, job, marital, education, default, housing, loan).
  Encode categorical variables with pd.get_dummies().
  Convert boolean dummies to numeric.
#### Baseline Models
  Train KNN, Logistic Regression, Decision Tree, and SVM on raw one-hot features.
  Evaluate using Accuracy and F1 Score.
#### Improved Models
  Apply feature scaling with StandardScaler.
  Use class_weight="balanced" to handle imbalance.
  Tune key hyperparameters.
  Re-evaluate models.
#### Visualization
  Compare Baseline vs Improved models with Seaborn bar charts for Accuracy and F1.

### Results

#### Baseline Models:
  High accuracy (~85–87%) but very low F1 (≈0.0–0.13).
  Models biased toward majority class (no).
#### Improved Models:
  Accuracy decreased (~60%) but F1 improved significantly (~0.26).
  Models better capture the minority class (yes), which is critical for the business objective.
#### Key Insight
  F1 is more important than Accuracy in this context.
  Improved models are more aligned with the bank’s business goal of identifying subscribers.
  
### Visualizations
Classifier Accuracy: Baseline vs Improved
Classifier F1 Score: Baseline vs Improved
(Generated using Seaborn).

### Next Steps
  Add campaign and economic context features for richer models.
  Use advanced imbalance handling techniques (e.g., SMOTE).
  Apply cross-validation and grid search for robust hyperparameter tuning.
  Explore ensemble methods (Random Forest, XGBoost).
  
### Tech Stack
  Python
  pandas, numpy
  scikit-learn
  seaborn, matplotlib
