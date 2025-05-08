#!/usr/bin/env python
# coding: utf-8

# # [프로젝트2] 작성한 코드 모듈화 및 테스트하기

# ## 프로젝트 목표
# ---
# - Jupyter Notebook 으로 완성한 코드를 클래스화(모듈화) 하고, 이를 테스트

# ## 프로젝트 목차
# ---
# 
# 1. **데이터 읽기:** 철근 가격 데이터 읽기 및 DataFrame 생성
# 
# 2. **데이터셋 생성:** src 라이브러리의 Datasets 클래스로 데이터셋 생성
# 
# 3. **모델 훈련:** train() 함수로 훈련
# 
# 4. **훈련 결과 시각화:** 모델의 훈련(손실) 결과 시각화
# 
# 5. **검증 및 시각화:** Data Loader로 검증 결과 도출 및 결과 시각화

# ## 1. 데이터 읽기

# ### 1.1 필요한 라이브러리 import

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import src
import warnings

# 라이브러리 reload
importlib.reload(src)

# Unicode warning 제거 (폰트 관련 경고메시지)
plt.rcParams['axes.unicode_minus']=False

# 한글 폰트 설정
# plt.rcParams['font.family'] = "AppleGothic"
plt.rcParams['font.family'] = "NanumGothic"

# 경고 메시지 출력 안함
warnings.filterwarnings('ignore')

# DataFrame 소수점 이하 4째자리까지 표현
pd.set_option('display.float_format', lambda x: f'{x:.4f}') 


# ### 1.2 데이터 불러오기

# 데이터는 '/mnt/elice/dataset' 에 위치해있으며, 파일명은 '전처리Data.xlsx' 입니다. (파일명에서도 나타나있듯, xlsx 확장자를 갖습니다.)

# In[2]:


# 데이터 읽기 (DataFrame)
df = pd.read_excel(r'/mnt/elice/dataset/전처리Data.xlsx')

# Shape 확인
print(df.shape)
df.head()


# ## 2. DataSets로 데이터셋 생성

# In[3]:


# DataSets 생성
dataset = src.DataSets(df, '철근 고장력 HD10mm [한국(출고가)] 현물KRW/ton', batch_size=32, window_size=6)


# In[4]:


# train, test 데이터 로더
train_loader = dataset.get_train_loader()
test_loader = dataset.get_test_loader()


# In[5]:


# 1개 배치 추출
x, y = next(iter(train_loader))


# In[6]:


# 추출된 데이터 shape
x.shape, y.shape


# ## 3. 모델 훈련

# In[7]:


import torch

# device 설정 (cuda 혹은 cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[8]:


losses, model = src.train(train_loader, num_epochs=1000, device=device)


# In[15]:


# 모델의 하이퍼파라미터 변경
model2 = src.LSTMModel(input_size=1, hidden_size=64, output_size=1, num_layers=2, bidirectional=True)


# In[16]:


# 재학습
losses2, model2 = src.train(train_loader, num_epochs=1000, device=device, model=model2)


# ## 4. 훈련 결과 시각화

# In[9]:


# 학습/검증 손실 시각화
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 4)
ax.plot(losses, color='red', linewidth=1.0, label='Training Loss')
ax.set_title('Losses')
ax.legend()
plt.show()


# ## 5. 검증 및 시각화

# In[10]:


# test_loader 에 대한 추론
y_trues, preds = src.evaluate(test_loader, model, dataset.get_scaler())


# In[11]:


# 예측, 실제 값 비교 시각화
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 4)
ax.plot(preds, color='tomato', linewidth=1.0, label='Prediction', linestyle='-.')
ax.plot(y_trues, color='skyblue', linewidth=1.0, label='Actual')
ax.set_title('Prediction vs Actual on 45 days')
ax.legend()
plt.show()


# In[12]:


# train_loader 에 대한 추론
y_trues, preds = src.evaluate(train_loader, model, dataset.get_scaler())


# In[13]:


# 예측, 실제 값 비교 시각화
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 4)
ax.plot(preds, color='tomato', linewidth=1.0, label='Prediction', linestyle='-.')
ax.plot(y_trues, color='skyblue', linewidth=1.0, label='Actual')
ax.set_title('Prediction vs Actual on Training Data')
ax.legend()
plt.show()

