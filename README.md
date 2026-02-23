# CTR Prediction Performance Improvement (XGBoost)

## Overview

본 프로젝트는 광고 클릭 여부(CTR)를 예측하는 이진 분류 문제입니다.  
공모전에서 제공한 **LSTM 기반 베이스라인 모델(AUC 0.6331)**을 분석한 뒤,  
정형(Tabular) 데이터에 더 적합한 **XGBoost + 5-Fold Cross Validation** 전략으로 모델을 개선했습니다.

---

## Results

| Model | Validation AUC |
|------|-----------------|
| LSTM Baseline | 0.6331 |
| XGBoost (5-Fold CV) | 0.71+ |

**Improvement: +0.07 ~ +0.08 AUC**

✔ 베이스라인 대비 약 11~13% 성능 향상  
✔ 모델 변경 + Feature Engineering + CV 전략을 통해 안정적인 일반화 성능 확보

---

## Problem Definition

- **Task:** CTR (Click-Through Rate) 예측 (Binary Classification)
- **Metric:** AUC (ROC-AUC)
- **Goal:** 공모전 제공 베이스라인 모델 대비 성능 개선

베이스라인은 LSTM + MLP 구조였으나,  
CTR 데이터는 순차 데이터보다는 정형 피처 중심 구조였기 때문에  
모델 선택 자체를 재검토했습니다.

---

## Why XGBoost?

- CTR 데이터는 **Structured Tabular Data**
- 트리 기반 앙상블 모델이 정형 데이터에서 높은 성능
- 자동 피처 상호작용 학습 가능
- Deep Learning 대비 안정적이고 빠른 학습

---

## Exploratory Data Analysis (EDA)

### 1) Target Distribution
- 클릭 비율 약 10~15%
- 불균형 데이터셋 확인
- → `scale_pos_weight` 적용

### 2) CTR by Hour
- 특정 시간대(저녁 시간대) CTR 상승 패턴
- → `hour_group` 파생변수 생성

### 3) CTR by Day of Week
- 요일별 CTR 편차 존재
- → `day_group` 파생변수 생성

### 4) Inventory CTR Variance
- `inventory_id` 별 CTR 편차 매우 큼
- → Target Encoding 적용

### 5) History Feature Analysis
- 일부 `history_*` 피처와 클릭 여부 간 높은 상관관계
- → `history_sum`, `history_mean`, `history_max` 생성

---

## Key Improvements

- LSTM → XGBoost 모델 변경
- Stratified 5-Fold Cross Validation 적용
- Target Encoding 도입
- 시간 기반 파생변수 추가
- History 집계 피처 생성
- 하이퍼파라미터 튜닝

---

## Experimental Setup

- 5-Fold Stratified Cross Validation
- AUC 기준 모델 평가
- Early Stopping 적용
- 주요 튜닝 파라미터:
  - learning_rate
  - max_depth
  - n_estimators
  - subsample
  - colsample_bytree
  - gamma

---

## Project Structure

```
src/
 ├── config.py          # 실험 설정/파라미터
 ├── data.py            # 데이터 로드
 ├── preprocess.py      # 전처리
 ├── features.py        # Feature Engineering
 ├── eda.py             # EDA 분석
 ├── train_xgb.py       # 단일 학습
 ├── train_xgb_cv.py    # 5-Fold CV 학습
 ├── train_xgb_full.py  # 전체 학습 + 제출 파일 생성
 └── utils.py           # 공통 유틸 함수
```

---

## How to Run

### 1️⃣ 데이터 위치

```
data/train.parquet
data/test.parquet
```

### 2️⃣ 패키지 설치

```bash
pip install -r requirements.txt
```

### 3️⃣ EDA 실행

```bash
python -m src.eda
```

### 4️⃣ 5-Fold CV AUC 확인

```bash
python -m src.train_xgb_cv
```

### 5️⃣ 전체 학습 및 제출 파일 생성

```bash
python -m src.train_xgb_full
```

출력 파일:
```
outputs/submission_xgb.csv
```

---

## Tech Stack

- Python
- XGBoost
- Scikit-learn
- Pandas / Polars
- PyArrow

---

## Conclusion

LSTM 기반 베이스라인 모델은 정형 CTR 데이터에 적합하지 않았으며  
모델 선택을 재검토한 결과, XGBoost가 더 높은 성능을 보였습니다.

✔ 모델 선택의 중요성  
✔ Feature Engineering의 영향력  
✔ Cross Validation 기반 안정적인 검증 전략  

을 통해 **AUC +0.07~0.08 개선**을 달성했습니다.
