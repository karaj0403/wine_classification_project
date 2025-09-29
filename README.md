# wine_classification_project
ReRquired capstone component 5.1: Analysing how data splitting affects the model's performance
# Wine Classification Project

This repository contains experiments with different training-validation-test splits for the Wine dataset using Logistic Regression and scaling.

## Project Overview
- Dataset: `sklearn.datasets.load_wine()`
- Models: Logistic Regression
- Preprocessing: StandardScaler
- Objective: Understand the impact of different data splits (80:20:20, 70:15:15, 60:20:20) on model performance.

## Experiments
1. 80:20:20 split (initial idea)
2. 70:15:15 split
3. 60:20:20 split

For each split:
- Model is trained on training set.
- Validation set used for hyperparameter tuning.
- Test set used for final evaluation.
- Accuracy and classification reports recorded.
- Confusion matrices generated for visualization.

## Usage
1. Install requirements:
scikit-learn
matplotlib
numpy
pandas
jupyter
