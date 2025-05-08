#!/usr/bin/env python
# coding: utf-8

# # 전기차 배터리 사용량 예측
# 

# ---

# B회사 전기차 연구 개발 부서에서는 최근 전기차 배터리의 성능 개선에 초점을 맞추고 있다. 이 과정에서, 전기차의 운행 데이터를 바탕으로 배터리 사용량을 보다 정확하게 예측할 수 있는 모델의 개발이 필요하다고 판단되었다.
# 
# 연구 개발 부서는 문제 해결 방안을 모색하기 위해 데이터 과학(DS) 팀에 연락하여, 전기차 운행 데이터를 활용해 배터리 사용량을 예측할 수 있는 모델 개발을 요청하였다.
# 
# 이제 당신은 DS 팀의 일원으로서, 전기차 운행 데이터를 활용하여 배터리 사용량 예측 모델을 개발해야 한다.

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

# In[1]:


get_ipython().system('pip list')


# 이하의 과제를 풀기 위하여 아래와 같은 라이브러리를 활용해 볼 수 있다.

# In[2]:


import os
import copy
import math
import json
import pickle
import random
import sklearn
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


# 필요하다면 [파이썬 표준 라이브러리](https://docs.python.org/3/library/index.html) 내 모듈이나 설치된 라이브러리를 임의로 불러올 수 있다.

# In[3]:


import itertools, functools, collections # 기타 필요한 모듈 ...


# ## 목차
# 
# **Ⅰ. 데이터 준비** (문제 1-1, 1-2, 1-3)
#     
# **Ⅱ. 머신러닝 모델 수행** (문제 2-1, 2-2)

# # Ⅰ. 데이터 준비

# ## [문제 1-1]

# ### [데이터 설명 1-1]
# 전달받은 데이터는 69회에 걸친 차량 주행 기록으로, 세 종류의 파일 (총 70개) 에 나뉘어 있다.
# 
# 세 종류의 파일은 모두 여러 행과 열로 구성된 정형 데이터이다.
# 
# 1. *metadata.csv*: 메타데이터 파일
# 
# 2. *TripA01.csv* ~ *TripA32.csv*: 32개의 여름철 주행 기록 파일
# 
# 3. *TripB01.csv* ~ *TripB37.csv*: 37개의 겨울철 주행 기록 파일

# ### [상황설명 1-1]

# 위 파일 이름 정보를 활용하여 데이터를 불러온다.

# ### [문제 설명 1-1] 데이터를 불러오시오.
# - 조건 1. 세 종류의 csv 파일을 pandas dataframe 형식으로 읽어온다.
# - 조건 2. 불러온 파일은 딕셔너리 변수인 `data`에 저장한다.
# - 조건 3. 해당 파일의 이름(.csv는 제외)을 key로 사용하여 `data`에 저장한다.
# (예를 들어, TripA01.csv 파일을 TripA01.csv인 경우 `data['TripA01']`로 저장)

# In[4]:


data_root = '/mnt/elice/dataset/data'
data = dict()
# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #
filenames = ['metadata']
for idx in range(1, 32 + 1): filenames.append('TripA{:02d}'.format(idx))
for idx in range(1, 37 + 1): filenames.append('TripB{:02d}'.format(idx))
for fn in filenames: data[fn] = pd.read_csv(os.path.join(data_root, '{}.csv'.format(fn)))


# ## [문제 1-2]

# ### [데이터 설명 1-2]

# 주행기록은 (Trip으로 시작하는 파일) 크게 네 가지 종류의 피처로 구성되어 있다.
# 
# *   환경 관련 피처 : ambient temperature(외기 온도), elevation(고도) 등
# *   자동차 관련 피처 : velocity(속도), throttle(가속 조절 장치), regenerative braking(회생 제동) 등
# *   배터리 관련 피처 : voltage(전압), current(전류), SoC(state of charge, 충전량) 등
# *   난방 회로 관련 피처 : cabin temperature(실내온도), coolant temperature(냉각수 온도), heating power(난방출력) 등

# ### [상황설명 1-2]

# 불러온 주행기록 데이터마다 가지고 있는 피쳐의 종류와 수가 일치하지 않는 것을 확인할 수 있다. 각 주행기록을 살펴보고, 모든 주행기록에 공통으로 존재하는 열만 피쳐로 사용하려고 한다.

# In[6]:


data['TripA01'].head()


# In[7]:


data['TripB01'].head()


# ### [문제 설명 1-2] Feature selection을 진행하시오.
# - 조건 1. 모든 주행기록(이름이 Trip으로 시작하는 파일)에서 공통으로 존재하는 피쳐만 사용한다.

# In[8]:


# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #

selected_cols = set(data['TripA01'].columns.tolist())
for key in filenames[1:]: selected_cols = set.intersection(selected_cols, set(data[key].columns.tolist()))
selected_cols = sorted(list(selected_cols))
for key in filenames[1:]: data[key] = data[key][selected_cols]


# ## [문제 1-3]

# ### [상황 설명 1-3]

# 사용하지 않을 피쳐들을 제거한 후에도 결측치가 존재하는 것을 확인할 수 있다. 결측치를 적절히 처리하는 작업을 수행한다.

# In[33]:


data['TripB08'].isnull().values.sum()


# ### [문제 설명 1-3] 결측치를 처리하시오.
# - 조건 1. 각 주행 기록에서 결측치의 개수가 5개 이상인 경우가 있을 시, 그 주행 기록은 분석 대상에서 제외한다.
# - 조건 2. 특정 부분의 결측치가 존재할 시에는 바로 직전 시간의 관측값으로 대체한다.
# - 조건 3. 바로 직전의 관측값이 없을 경우, 바로 직후의 관측값으로 대체한다.

# In[37]:


# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #

removed_keys = set()
data_keys = list()

for idx in range(1, 32 + 1): data_keys.append('TripA{:02d}'.format(idx))
for idx in range(1, 37 + 1): data_keys.append('TripB{:02d}'.format(idx))

for key in data_keys:
    for col in data[key].columns:
        if data[key][col].isnull().values.any():
            if data[key][col].isnull().values.sum() >= 5: removed_keys.add(key)
            data[key][col] = data[key][col].fillna(method = 'ffill').fillna(method = 'bfill')

for key in removed_keys:
    data.pop(key)

# 전체 데이터 키 중 제거된 키를 제외한 나머지 Key를 저장한다.
selected_keys = sorted(list(set(data_keys) - removed_keys))


# # Ⅱ. 머신러닝 모델 수행

# ## [문제 2-1]

# ### [상황 설명 2-1]

# 모델링을 위해 데이터를 한번 더 가공해준다. 우선 주행기록 데이터의 변화량을 1분 단위로 구한다.

# In[43]:


tick = 600

for key in list(data.keys()):
    if key == 'metadata':
        pass
    else:
        trip = data[key].copy()
        delta = trip.diff(periods = tick).iloc[tick::tick]
        data['{}_delta'.format(key)] = delta.copy()


# 주행기록 데이터의 평균값을 1분 단위로 구해준다.

# In[17]:


tick = 600

for key in selected_keys:
    trip = data[key].copy()
    moving = trip.rolling(window = tick).mean().iloc[tick::tick]
    data['{}_moving'.format(key)] = moving.copy()


# 분석을 위한 피쳐는 평균값을 이용하고, 예측하고 싶은 배터리 사용량은 변화량을 이용하여 데이터를 완성한다.

# In[45]:


loaded = np.load('/mnt/elice/dataset/preprocessed_data.npz')
train_x, train_y = loaded['train_x'], loaded['train_y'].ravel()
test_x, test_y = loaded['test_x'], loaded['test_y'].ravel()


# 우리는 위의 데이터를 이용하여 회귀 분석을 진행하려고 한다.

# ### [문제 설명 2-1] 머신러닝 모델링을 진행하시오. (6점)
# 
# - 조건 1. GBM 모델을 사용한다.
# - 조건 2. Scikit-learn의 GradientBoostingRegressor를 사용한다.
# - 조건 3. GradientBoostingRegressor의 random_state는 3064로 설정하여 사용한다.

# In[46]:


# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #

my_model = GradientBoostingRegressor(random_state=3064)
my_model.fit(train_x, train_y)

train_p = my_model.predict(train_x)
test_p = my_model.predict(test_x)


# In[49]:


def mae_error(pred: np.ndarray,
              gt: np.ndarray) -> float:
    mae = float(np.abs(pred - gt).mean())
    return mae

train_mae = mae_error(train_p, train_y)
test_mae = mae_error(test_p, test_y)

print(train_mae, test_mae)


# ## [문제 2-2]

# ### [상황 설명 2-2]

# GBM 모델을 최적화 시켜주기 위해 하이퍼파라미터 서치를 진행한다.

# ### [문제 설명 2-2] 하이퍼파라미터 튜닝을 진행하시오. (6점)
# 
# - 조건 1. Grid Search 방식을 사용하시오.
# - 조건 2. GradientBoostingRegressor의 `max_depth`, `learning_rate`, `n_estimators`를 튜닝하시오.
# - 조건 3. 하이퍼파라미터의 후보들은 아래에 정의된 `MAX_DEPTH`, `LEARNING_RATE`, `N_ESTIMATORS`를 활용하시오.
# - 조건 4. GradientBoostingRegressor의 random_state는 항상 3064로 고정하시오.

# In[50]:


seed = 3064
MAX_DEPTH = [2, 3, 4]
LEARNING_RATE = [0.1, 0.05, 0.01]
N_ESTIMATORS = [60, 80, 100, 120]


# In[51]:


# ================================================================== #
#         START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)        #
# ================================================================== #
#
#                이 곳에서 코드를 수정 및 작성하시오.
#
# ================================================================== #
#          END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)         #
# ================================================================== #

best_hyperparameter = None
best_model = None
best_mae = 1000000

hyperparameters = list(itertools.product(MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS))
for max_depth, lr, n_estimators in hyperparameters:

    model = GradientBoostingRegressor(max_depth = max_depth,
                            learning_rate = lr, n_estimators = n_estimators, random_state = seed)
    model.fit(train_x, train_y)
    test_p = model.predict(test_x)
    test_mae = mae_error(test_p, test_y)

    if best_mae > test_mae:
        best_hyperparameter = (max_depth, lr, n_estimators)
        best_model = copy.deepcopy(model)
        best_mae = test_mae


# In[52]:


print('[info] Hyperparameter tuning finished')
print('[info] Best hyperparameters: {}'.format(best_hyperparameter))
print('[info] Best test mae: {}'.format(best_mae))


# ---
