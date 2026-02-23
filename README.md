# CTR Prediction Performance Improvement

## Overview
This project improves CTR prediction performance by replacing an LSTM baseline model with XGBoost and applying 5-Fold Cross Validation.

## Results

| Model | AUC |
|--------|------|
| LSTM Baseline | 0.6331 |
| XGBoost (5-Fold CV) | 0.71+ |

Improvement: +0.07~0.08 AUC

## Key Improvements
- Replaced LSTM with XGBoost for tabular CTR data
- Applied 5-Fold Cross Validation
- Hyperparameter tuning
- Feature-based optimization

## Tech Stack
- Python
- XGBoost
- PyTorch
- Scikit-learn
