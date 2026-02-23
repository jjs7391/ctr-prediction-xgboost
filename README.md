# CTR Prediction Performance Improvement

## Overview
This project improves CTR prediction performance by replacing an LSTM baseline model with XGBoost and applying 5-Fold Cross Validation.

## Exploratory Data Analysis (EDA)

### Target Distribution
- 클릭 비율은 약 10~15% 수준으로 불균형 데이터셋임을 확인
- → scale_pos_weight 적용

### CTR by Hour
- 특정 시간대(저녁 시간대)에서 CTR이 높게 나타남
- → hour_group 파생변수 생성

### CTR by Day of Week
- 요일에 따른 클릭 편차 존재
- → day_group 변수 생성

### Inventory CTR Variance
- inventory_id별 CTR 편차가 매우 큼
- → Target Encoding 적용

### History Feature Correlation
- 일부 history feature가 클릭 여부와 높은 상관관계를 보임
- → history 기반 파생변수 추가

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

실행방법

data/train.parquet

data/test.parquet

pip install -r requirements.txt

# CV AUC 확인
python -m src.train_xgb_cv

# 제출파일 생성(outputs/submission_xgb.csv)
python -m src.train_xgb_full
