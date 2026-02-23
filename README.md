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

## Problem Definition

The baseline LSTM model showed limited performance (AUC 0.6331) for tabular CTR prediction data.
Given the nature of structured tabular features, we hypothesized that tree-based ensemble models would outperform sequence-based deep learning models.

## Why XGBoost?

- CTR dataset is structured tabular data
- Tree-based models are known to perform strongly on tabular problems
- Handles feature interactions effectively
- More stable training compared to deep learning baseline

- ## Experimental Setup

- 5-Fold Cross Validation
- Stratified data split
- AUC as evaluation metric
- Hyperparameter tuning (learning rate, max depth, n_estimators)

- ## Conclusion

Replacing the LSTM baseline with XGBoost significantly improved performance (+0.07~0.08 AUC).
This demonstrates the importance of selecting models appropriate for structured tabular data.
