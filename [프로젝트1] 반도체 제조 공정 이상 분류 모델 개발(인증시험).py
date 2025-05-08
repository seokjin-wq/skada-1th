#!/usr/bin/env python
# coding: utf-8

# # 반도체 제조 공정 이상 분류 모델 개발

# <img src="images/image_1.jpeg" width="1000"/>

# A회사 반도체 제조 현장 담당자는 최근 반도체 제조 공정에서 불량률이 급증하고 있다는 사실을 발견했다. 
# 
# 현장 담당자는 문제 해결 방안을 찾기 위해 DS부서에 공정 데이터를 바탕으로 제품의 양/불량을 조기에 분류할 수 있는 모델 개발을 요청해왔다. 
# 
# 이제 당신은 DS 담당자로서, 반도체 제조 공정 데이터를 활용하여 이상 분류 모델을 개발하여야 한다.

# ## 시험 주의 사항
# 
# - 아래와 같이 **주석으로 구분된 곳에서 코드를 수정 및 작성** 해야하며, 표시된 곳 **이외의 코드 및 하이퍼파라미터 값을 임의로 수정하면 오답 처리 된다.**
# 
# ```python
#         # ================================================================== #
#         #         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
#         # ================================================================== #
#         #
#         #                이 곳에서 코드를 수정 및 작성하시오.
#         #
#         # ================================================================== #
#         #          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
#         # ================================================================== #
# ```
# 
# - 실제 SKADA 시험에서는 각 문제마다 **\[답안 검증\] 및 \[답안 제출\]** 셀이 제공된다.
#     - **\[답안 검증\]** : 문제에 대한 답안을 중간 점검하기 위한 **최소 조건을 제공**하는 것으로, 실제 정답을 맞췄는지 여부와는 다를 수 있다.
#     - **\[답안 제출\]** : 답안 작성 후 **반드시 \[답안 제출\] 셀을 실행**시켜 `skada` 객체로 답을 제출해야 정답으로 인정된다. **답안제출 코드를 임의로 수정하면 정상 제출을 보장하지 않는다**.

# ## 라이브러리 불러오기
# 현재 환경에서 사용할 수 있는 외부 라이브러리는 다음과 같다.

# In[ ]:


get_ipython().system('pip list')


# 이하의 과제를 풀기 위하여 아래와 같은 라이브러리를 활용해 볼 수 있다.

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import random
import xgboost as xgb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , precision_score , f1_score, roc_curve, auc, roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE

from typing import Any
import joblib

from utils import visualize_top_features_pca, visualize_pca_loadings

import utils
skada = utils.SKADA.instance()

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(42)


# 필요하다면 [파이썬 표준 라이브러리](https://docs.python.org/3/library/index.html) 내 모듈이나 설치된 라이브러리를 임의로 불러올 수 있다.

# In[ ]:


import itertools, functools, collections # 기타 필요한 모듈 ...


# ## 목차
# 
# **Ⅰ. 데이터 준비** (문제 1-1, 1-2)
#     
# **Ⅱ. 모델 학습** (문제 2-1, 2-2)
#     
# **Ⅲ. 모델 고도화** (문제 3-1, 3-2)
# 
# **Ⅳ. 심화 문제** (문제 4)

# # Ⅰ. 데이터 준비

# ## [문제 1-1]

# ### [데이터 설명 1-1]
# 현장 담당자로부터 전달 받은 데이터의 상세 정보는 다음과 같다.
# 
# - 제조 공정명 : 반도체 제조 공정
# - 수집 장비 : 반도체 제조 설비 내 센서 데이터
# - 데이터셋 구조 : 피쳐 데이터는 테이블 형식(Tabular)로, 총 378개의 칼럼과 1637개의 샘플로 구성되어 있다. 라벨 데이터는 1637개의 값으로 구성되어 있다.

# Feature 변수 `x`와 label 변수 `y`가 갖고 있는 각 **칼럼의 이름**과 의미는 다음과 같다.

# - ##### 변수 `x`
# **0~377**: 반도체 제조 설비 내 센서 데이터로 측정된 익명화된 정보
# - ##### 변수 `y`
# **Pass/Fail**: 공정 과정 중 이상 샘플 테스트 통과 여부 (0: 통과, 1: 실패)

# ### [상황설명 1-1]

# 다음과 같이 데이터를 불러와 피쳐 `x`와 라벨 `y`를 설정한다.

# In[ ]:


x = pd.read_csv('/mnt/elice/dataset/feature_p1.csv')
y = pd.read_csv('/mnt/elice/dataset/label_p1.csv')


# In[ ]:


print(x.shape)
print(y.shape)


# feature 변수 `x`은 샘플 수에 비해 많은 feature를 가지고 있다.
# 
# PCA는 고차원 데이터에서 주요한 정보를 유지하면서 차원을 줄이는 기법으로, 데이터의 패턴을 더 쉽게 이해하거나, 계산 효율성을 높이기 위해 사용된다. 
# 
# Feature의 수를 줄이기 위해 feature `x`에 PCA를 적용하여 데이터의 차원을 축소한다.
# 
# 또한 그 결과를 바탕으로 데이터의 패턴을 이해하기 위해 2차원 시각화를 진행한다.
# 
# 시작하기 전에, train data와 test data를 구분하고, 각 feature의 평균이 0이고 표준편차가 1이 되도록 데이터를 표준화 (standardization)한다. 표준화는 PCA의 성능을 최적화하기 위해 필요하다.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train[x_train.columns] = scaler.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scaler.transform(x_test[x_test.columns])


# ### [문제 설명 1-1] 데이터의 차원을 축소하시오. (6점)
# - 조건 1. `x_train`과 `x_test`의 차원을 축소한다.
# - 조건 2. `scikit-learn`의 PCA 함수를 사용한다. (Full 함수 이름)
# - 조건 3. 원본 데이터의 분산 정보 중 95% 이상 유지하는 최적의 차원을 찾는다. 즉, 95% 이상 유지가 되는 차원의 후보들 중 가장 작은 값의 차원으로 PCA를 적용한다.
# - 조건 4. 조건 3의 최적의 차원 값을 `d`로 저장한다.
# - 조건 5. PCA 학습은 `x_train`만 사용한다.
# - 조건 6. 결과 변수명은 `x_train_pca`와 `x_test_pca`이다.

# In[ ]:


# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #


# ### [답안 검증 1-1]

# In[ ]:


assert np.sum(pca.explained_variance_ratio_) >= 0.95
assert x_train_pca.shape[1] == x_test_pca.shape[1]
assert np.allclose(pca.mean_, x_train.mean(axis=0))


# ### [답안 제출 1-1]  수정 금지, 실행만 하시오.

# In[ ]:


skada.Q1_1_answer(x_train_pca, x_test_pca, d)


# ## [문제 1-2]

# ### [문제 설명 1-2] 데이터 시각화를 진행하시오. (6점)
# 
# - 조건 1. PCA의 결과를 이용한다.
# - 조건 2. 학습 데이터를 이용하여 시각화 한다.
# - 조건 3. 원본 데이터의 분산에 대한 정보를 가장 많이 갖고 있는 두개의 주성분 벡터를 이용하여 2차원 시각화를 진행한다.
# - 조건 4. 라이브러리는 matplotlib과 seaborn 모두 사용 가능하다.
# - 조건 5. Scatter plot으로 데이터를 표현한다.
# - 조건 6. 포인트의 색은 label에 따라 다르게 한다.

# **참고**: PCA를 통한 2차원 시각화는 데이터의 주요 패턴을 간략하게 보여주는 것이 목적이다. 그러나 이 방법만으로 항상 클래스를 명확하게 구분할 수 있는 것은 아니다. 시각화 결과가 두 클래스 간에 명확한 구분을 보여주지 않더라도, 이는 선형 변환인 PCA의 한계일 수 있으며 데이터의 실제 특성과 관련이 있을 수 있다.

# In[ ]:


# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #

plt.xlabel(f'First Principal Component')
plt.ylabel(f'Second Principal Component')
plt.title('2D PCA of Data with Top 2 Principal Components')
plt.show()


# ### [답안 검증 1-2]

# 만약 데이터 시각화를 올바르게 했다면, 아래와 같은 형태의 그래프를 얻을 수 있다. 단, 결과가 아래 그래프와 완전히 일치할 필요는 없으며, 문제의 요구사항과 조건을 충족하는 그래프라면 정답으로 간주된다.

# <img src="images/2-2.jpg" width="800"/>

# # Ⅱ. 모델 학습

# ## [문제 2-1]

# ### [상황 설명 2-1]

# 지금까지의 작업으로 다음과 같은 결과를 얻을 수 있었다.

# In[ ]:


x_train_pca = np.load('/mnt/elice/dataset/train_feature_2_p1.npy')
x_test_pca = np.load('/mnt/elice/dataset/test_feature_2_p1.npy')


# 데이터가 준비되었으니, 본격적으로 학습 모델을 구현해보자. 우리는 XGBoost를 이용하여 binary classification model을 학습한다. 우선 현재 처리되어 있는 상태의 데이터들을 이용하여 학습 및 평가를 진행한다. 본 프로젝트에서는 ROC-AUC 값을 통해 평가를 진행한다.

# ※ 모델이 정상적으로 학습하는 경우 **3분** 이내에 학습이 완료되는 것을 확인 할 수 있다.

# In[ ]:


model = XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=200)
model.fit(x_train_pca, y_train)
y_pred = model.predict(x_test_pca)

print("Basic XGBoost Performance:")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("ROC-AUC: ", roc_auc_score(y_test, y_pred))


# 현재 학습한 모델의 결과는 accuracy가 높지만 roc-auc는 만족스럽지 않은 상태이다. 우선 데이터의 불균형을 처리함으로써 성능 향상을 시도한다.

# ### [이론 설명 2-1]
# 데이터 불균형은 모델의 성능에 부정적인 영향을 줄 수 있다. SMOTE(Synthetic Minority Over-sampling Technique)는 불균형한 클래스 분포를 가진 데이터셋에서 소수 클래스의 샘플을 합성하여 데이터 불균형 문제를 완화하는 방법이다. 
# 
# <img src="images/smote.jpg" width="400"/>
# 
# ### SMOTE 동작 과정:
# 
# 1. **Minority Class Sample 선택**: 소수 클래스에서 랜덤하게 하나의 인스턴스를 선택한다.
# 2. **k-Nearest Neighbors 계산**: 선택된 인스턴스와 가장 가까운 거리에 있는 k개의 소수 클래스 샘플을 찾는다.
# 3. **Synthetic Sample 생성**: 선택된 인스턴스와 k개의 이웃 중 하나 사이의 선형 보간을 통하여 새로운 인스턴스를 생성한다.

# ### [문제 설명 2-1] 데이터 불균형을 처리하시오. (4점)
# 
# - 조건 1. SMOTE를 활용하여 불균형을 처리한다. (imblearn 라이브러리 사용 가능)
# - 조건 2. 결과 변수명은 `x_train_pca`, `y_train`에서 `x_resampled`, `y_resampled`로 변경한다.

# In[ ]:


# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #


# ### [답안 검증 2-1]

# In[ ]:


class_counts = y_resampled.value_counts()
assert class_counts[0] == class_counts[1]


# ## [문제 2-2]

# ### [상황 설명 2-2]

# 지금까지의 작업으로 다음과 같은 결과를 얻을 수 있었다.

# In[ ]:


x_resampled = np.load('/mnt/elice/dataset/feature_3_p1.npy')
y_resampled = pd.read_csv('/mnt/elice/dataset/label_3_p1.csv')


# SMOTE를 적용한 데이터셋을 이용하여 다시 한 번 모델의 학습 및 평가를 진행한다. 

# ※ 모델이 정상적으로 학습하는 경우 **3분** 이내에 학습이 완료되는 것을 확인 할 수 있다.

# In[ ]:


model = XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=200)
model.fit(x_resampled, y_resampled)
y_pred = model.predict(x_test_pca)

print("Basic XGBoost Performance:")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("ROC-AUC: ", roc_auc_score(y_test, y_pred))


# ### [문제 설명 2-2]최적의 하이퍼파라미터를 찾아보시오. (4점)
# 
# 하이퍼파라미터는 모델의 성능에 큰 영향을 줄 수 있으며, Grid Search는 가능한 모든 조합을 시도하여 최적의 하이퍼파라미터를 찾는 방법이다.
# 
# - 조건 1. `XGBClassifier`의 하이퍼파라미터를 찾는다.
# - 조건 2. `GridSearchCV` 함수를 사용한다. (`sklearn.model_selection.GridSearchCV`)
# - 조건 3. `param_grid` 변수에 저장된 값들을 하이퍼파라미터 후보로 사용한다.
# - 조건 4. 하이퍼파라미터 선택 기준은 ROC-AUC가 높은 값을 기준으로 한다.
# - 조건 5. 3-fold cross validation을 사용한다.
# - 조건 6. GridSearchCV 객체의 변수 이름은 `grid_search`로 설정한다.

# ※ 모델이 정상적으로 학습하는 경우 **5분** 이내에 grid search가 완료되는 것을 확인 할 수 있다.

# In[ ]:


param_grid = {
    'max_depth': [4, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'gamma': [0.1, 0.5]
}

# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #
print("\nBest Hyperparameters:", grid_search.best_params_)


# ### [답안 검증 2-2]

# In[ ]:


for param, value in grid_search.best_params_.items():
    assert value in param_grid[param]


# # Ⅲ. 모델 고도화

# ## [문제 3-1]

# ### [상황 설명 3-1]
# 현재 모델의 성능보다 더 높은 성능을 갖도록 모델 고도화를 진행한다.
# 
# 데이터 분석 부서의 팀 리더는 모델의 다양성을 확보하고 여러 접근 방식을 탐색하기 위해 팀원들에게 다양한 모델 고도화 기법을 아래와 같이 수행하도록 했다.
# - A 팀원에게는 랜덤 포레스트를 사용하여 모델을 구축하는 작업이 배정되었다.
# - B 팀원에게는 딥러닝 기반의 신경망 모델을 구축하는 작업이 배정되었다.
# 
# - **당신에게는 스태킹을 사용한 앙상블 모델링 작업이 배정되었다.**

# ### [이론 설명 3-1]

# 스태킹(Stacking)은 여러 개의 모델의 예측 결과를 입력으로 사용하여 새로운 메타 모델을 학습시키는 앙상블 기법이다. 베이스 모델은 원본 데이터로부터 예측을 수행하며, 메타 모델은 베이스 모델들의 예측 결과를 바탕으로 최종 예측을 수행한다. 이 방법은 다양한 모델의 강점을 결합하여 전체적인 성능을 향상시키는 데 도움을 준다.
# 

# ### [문제 설명 3-1] 스태킹 모델을 학습하시오. (6점)
# 
# - 조건 1. 베이스 모델로 열개의 XGBoost classifier를 사용한다. 모든 모델의 하이퍼파라미터는 아래와 같이 설정한다:
#     - gamma=0.5
#     - learning_rate=0.1
#     - max_depth=4
#     - n_estimators=20
# - 조건 2. 각 베이스 모델의 `random_state`는 41~50 중 서로 다른 값을 사용한다.
# - 조건 3. scikit-learn 라이브러리에 구현되어 있는 stacking classifier를 활용한다.
# - 조건 4. 메타 모델로는 Logistic regression을 사용한다.
# - 조건 5. StackingClassifier 객체의 변수 이름은 `stacking_clf`로 설정한다.

# ※ 모델이 정상적으로 학습하는 경우 **5분** 이내에 학습이 완료되는 것을 확인 할 수 있다.

# In[ ]:


# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #

y_pred_stack = stacking_clf.predict(x_resampled)
accuracy = accuracy_score(y_resampled, y_pred_stack)
print(f"Stacking Model Accuracy: {accuracy:.4f}")


# ### [답안 검증 3-1]

# In[ ]:


base_models = stacking_clf.estimators_
assert all(isinstance(model, XGBClassifier) for model in base_models)
assert isinstance(stacking_clf.final_estimator, LogisticRegression)


# ## [문제 3-2]

# ### [문제 설명 3-2] 메타 모델의 최적의 하이퍼파라미터를 찾으시오. (6점)
# - 조건 1. 스태킹 모델의 메타 모델의 하이퍼파라미터를 찾는다.
# - 조건 2. `GridSearchCV` 함수를 사용한다. (`sklearn.model_selection.GridSearchCV`)
# - 조건 3. `param_grid` 변수에 저장된 값들을 하이퍼파라미터 후보로 사용한다.
# - 조건 4. 하이퍼파라미터 선택 기준은 ROC-AUC가 높은 값을 기준으로 한다.
# - 조건 5. 3-fold cross validation을 사용한다.
# - 조건 6. `GridSearchCV` 객체의 변수 이름은 `grid_search`로 설정한다.

# ※ 모델이 정상적으로 학습하는 경우 **5분** 이내에 grid search가 완료되는 것을 확인 할 수 있다.

# In[ ]:


param_grid = {
    'final_estimator__C': [0.01, 0.1]
}

# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #

print("Best parameters: ", grid_search.best_params_)


# 하이퍼파라미터 서치로 찾아진 best parameter를 갖는 모델로 성능을 평가한다.

# In[ ]:


y_pred_stack = grid_search.predict(x_resampled)
accuracy = accuracy_score(y_resampled, y_pred_stack)
print(f"Stacking Model Accuracy with Best Parameters: {accuracy:.4f}")


# ### [답안 검증 3-2]

# In[ ]:


assert 'final_estimator__C' in grid_search.best_params_
assert grid_search.best_params_['final_estimator__C'] in param_grid['final_estimator__C']
assert grid_search.estimator == stacking_clf


# # IV. 심화 문제

# ### [상황 설명 4]
# 
# SKADA 시험에서는 참가자들의 역량을 평가하기 위한 심화 문제를 출제한다. 이전 시험에서는 머신러닝 모델을 설명하는 문제가 심화 문제로 주어졌다. <br>
# 모델의 결정에 대한 해석을 진행하기 위해 모델 설명 방법을 구현하여 적용해 본다.

# ### [문제 설명 4] LIME 구현하기 (8점)
# 
# LIME (Local Interpretable Model-agnostic Explanations)은 복잡한 머신러닝 모델의 예측을 설명하기 위한 방법 중 하나다. 이 문제에서는 LIME의 동작 방식을 단순화한 `SimpleLime` 클래스를 구현한다.
# 
# 이 문제에서 사용될 `SimpleLime`은 주어진 데이터 포인트 주변에서 작은 변화를 주어 새로운 샘플들을 생성하고, 이 샘플들에 대한 모델의 예측을 사용하여 간단한 선형 모델을 훈련시킨다. 이 선형 모델의 계수는 특정 데이터에 대한 원래 모델의 예측을 설명하는 데 사용되는 feature의 중요도를 나타낸다.
# 
# **`explain` 메서드 수도코드:**
# 1) `_generate_samples` 메서드를 사용하여 100개의 샘플들을 생성한다.
# 2) 1번에서 생성된 샘플을 분석 대상 모델의 입력으로 사용하여, class 1에 대한 예측값을 얻는다 (힌트 : `predict_proba` 함수 활용)
# 3) 이 예측값들을 데이터의 새로운 타겟으로 설정하여 새로운 선형 회귀 모델을 훈련시킨다.
# 4) 선형 모델의 계수를 특성의 중요도로 반환한다.
# 
# 아래 조건에 맞춰 SimpleLIME 클래스의 **explain 메서드**를 구현한다.
# 
# - 조건 1. `explain` 메서드에서 `SimpleLime`의 동작 코드를 완성한다.
# - 조건 2. 선형 회귀 모델로는 `LinearRegression`을 사용한다.
# - 조건 3. 데이터에 추가할 노이즈는 표준 정규 분포에서 샘플링하여 사용한다.

# In[ ]:


class SimpleLIME:
    def __init__(self, model: Any, num_samples: int = 100):
        """
        Initialize the SimpleLIME.
        
        Parameters:
        - model: The black-box model we want to explain.
        - num_samples: Number of perturbed samples to generate.
        """
        self.model = model
        self.num_samples = num_samples

    def _generate_samples(self, data_point: np.ndarray) -> np.ndarray:
        """
        Generate perturbed samples around a given data point.
        
        Parameters:
        - data_point: The data point around which to generate samples.
        
        Returns:
        - Perturbed samples.
        """
        noise = np.random.normal(loc=0, scale=1, size=(self.num_samples, data_point.shape[0]))
        samples = data_point + noise
        
        return samples

    def explain(self, data_point: np.ndarray) -> np.ndarray:
        """
        Explain the prediction of a given data point.
        
        Parameters:
        - data_point: The data point to explain.
        
        Returns:
        - feature_importances: Feature importances.
        """
        
        # ================================================================== #
        #         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
        # ================================================================== #
        #
        #                이 곳에서 코드를 수정 및 작성하시오.
        #
        # ================================================================== #
        #          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
        # ================================================================== #


# ### [답안 검증 4]

# 위에서 구현한 `SimpleLIME` 클래스를 사용하여 모델 해석 결과를 분석 및 시각화를 진행한다. 이때 테스트 데이터셋 중 오분류 된 데이터 중에서 가장 높은 컨피던스 값으로 오분류를 하고 있는 샘플 하나를 골라서 분석을 진행한다.

# In[ ]:


y_prob = grid_search.predict_proba(x_test_pca)[:, 1]
errors = np.abs(y_test.values.reshape(-1) - y_prob)
index_max_error = np.argmax(errors)

lime_explainer = SimpleLIME(grid_search)
feature_importances_error = lime_explainer.explain(x_test_pca[index_max_error])

error_index = visualize_top_features_pca(feature_importances_error, "Top Feature Importances for Sample with Maximum Error")
visualize_pca_loadings(pca, x_train.columns, np.argmax(np.abs(feature_importances_error)))


# # 주관식 문제 종료
