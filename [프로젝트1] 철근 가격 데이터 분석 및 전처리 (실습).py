# # [프로젝트1] 철근 가격 데이터 분석 
# ## 프로젝트 목표
# ---
# - 시계열 데이터 분석 및 시각화
# - 정규화, 윈도우 데이터셋 구축 등 시계열 데이터 전처리
# - 시계열 데이터 예측 모형 구축, 추론 및 결과 시각화

# ## 프로젝트 목차
# ---
# 1. **데이터 읽기:** 철근 가격 데이터 읽기 및 DataFrame 생성
# 
# 2. **데이터 정제:** Date 컬럼 index 지정. 주간 데이터 추출
# 
# 3. **데이터 시각화:** 변수 시각화를 통하여 분포 파악
# 
# 4. **데이터 전처리:** 딥러닝 모델에 필요한 입력값 형식으로 데이터 처리
# 
# 5. **딥러닝 모델 정의:** PyTorch 라이브러리를 활용하여 LSTM 레이어를 활용한 모델 정의 및 생성
# 
# 6. **딥러닝 모델 훈련:** 생성한 모델 훈련 및 검증
# 
# 7. **추론 및 시각화:** 미래 데이터 결과 추론 및 추론 결과 시각화

# ## 1. 데이터 읽기
# ---
# 프로젝트에 활용할 데이터를 `Pandas`의 `read_excel()` 함수로 불러옵니다

# ### 1.1 라이브러리 불러오기

# In[1]:

# 필요한 모듈 import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Unicode warning 제거 (폰트 관련 경고메시지)
# plt.rcParams==> 다양한 설정 파라미터 원하는 대로 조정 할 수 있는 함수!
# 이 코드는 일부 플롯에서 유니코드 마이너스 기호가 잘못 표시되는 문제 해결!!
plt.rcParams['axes.unicode_minus']=False

# 한글 폰트 설정
# plt.rcParams['font.family'] = "AppleGothic"
plt.rcParams['font.family'] = "NanumGothic"

# 경고 메시지 출력 안함
warnings.filterwarnings('ignore')

# DataFrame 소수점 이하 넷째자리까지 표현
# set_option 함수 -> 출력 옵션 설정하는데 사용!
# 'display.float_fomat -> DataFrame or Series의 소수점 이하 숫자의 포맷 지정
# lamda 함수 사용해서 x를 소수점 4자리 까지 표시
pd.set_option('display.float_format', lambda x: f'{x:.4f}') 


# ### 1.2 데이터 불러오기

# In[2]:
import glob
glob.glob("/mnt/elice/dataset/*")


# In[3]:
# 데이터 로드
df = pd. read_excel("/mnt/elice/dataset/전처리Data.xlsx")
# 데이터 형태
print(df.shape)
# 상위 10개 행 조회
df.head(10)




# ## 2. 데이터 정제
# - `Date` 컬럼을 index로 지정합니다.
# - 주간 데이터를 추출합니다.

# ### 2.1 Date 인덱스 지정하기
# #### **[TODO]** df Dataframe에 있는 'Date' 칼럼을 index로 지정해 주세요.

# In[4]:
# Date 컬럼을 index로 지정

# df Dataframe에 있는 'Date' 칼럼을 index로 지정해 주세요.
df=df.set_index("Date")
df.head()


# In[5]:
df.loc['2013-01-11']


# ### 2.2 주(Weekly) 단위의 데이터만 추출

# #### **[TODO]** 주간(Weekly) 데이터만 추출하기 위하여 date_range()를 주 단위로 생성합니다.
# 

# In[6]:
# 주간(Weekly) 데이터만 추출하기 위하여 date_range()를 주 단위로 생성합니다.
target_index = pd. date_range("20130107","20211227",freq="W-MON")
target_index


# In[7]:
# index가 target_index를 포함하는 경우만 추출합니다
df = df.loc[df.index.isin(target_index)]
df


# ### 2.3 철근 고장력 HD10mm [한국(출고가)] 현물KRW/ton 컬럼 선택
# 
# #### **[TODO]** 철근 고장력 HD10mm [한국(출고가)] 현물KRW/ton 컬럼 을 선택하여 series 변수에 추출합니다.
# - Hint. 해당 컬럼은 index 3 에 있습니다.    

# In[8]:
df.head(3)


# In[9]:
# df에서 철근 고장력 HD10mm [한국(출고가)] 현물KRW/ton 컬럼 을 선택하여 series 변수에 추출합니다.
# Hint. 해당 컬럼은 index 3 에 있습니다.
series = df.iloc[:, 3]
series




# ## 3. 데이터 시각화

# ### 3.1 가격 데이터 시각화

# In[10]:


# 가격 데이터 시각화
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 4)
ax.plot(series, color='#6100E0', linewidth=2.0, label='Price Data')
ax.set_title(series.name)
ax.legend(loc="upper left")
plt.show()


# In[11]:


series


# ## 4. 데이터 전처리
# ---
# - 가격 데이터 정규화
# - 윈도우 데이터셋 구성

# ### 4.1 가격 데이터 정규화

# #### **[TODO]** scikit-learn의 Min Max Scaler를 활용해서 `series` 변수를 정규화 및 변환 합니다.
# - `reshape` 함수를 활용하여 차원을 1D 에서 2D로 1차원 늘려줍니다. 

# In[12]:


# 정규화(Normalization) 수행
from sklearn.preprocessing import MinMaxScaler

# scikit-learn의 MinMaxScaler를 정의하여 scaler 변수에 할당합니다.
scaler = MinMaxScaler()

# series를 변환하되, 1D -> 2D 차원을 1차원 늘려준 뒤 변환합니다
scaled_series = scaler.fit_transform(series.values.reshape(-1,1))

# 5개 데이터 출력 (numpy array)
scaled_series[:5]


# In[13]:


pd.DataFrame(scaled_series).describe().ndim


# ### 4.2 시계열 데이터셋 구성 (Windowed Dataset)

# In[14]:


def make_dataset(series, window_size=6):
    # Xs: 학습 데이터, Ys: 예측 데이터
    Xs = []
    Ys = []
    for i in range(len(series) - window_size):
        Xs.append(series.iloc[i:i+window_size].values)
        Ys.append(series.iloc[i+window_size])
    return np.array(Xs), np.array(Ys)


# In[15]:


scaled_series.flatten().ndim


# In[16]:


pd.Series(scaled_series.flatten()).ndim


# #### **[TODO]** 데이터셋을 생성하는 코드를 작성합니다.

# In[17]:


# 데이터셋을 생성하는 코드를 작성합니다.
Xs, Ys = make_dataset(pd.Series(scaled_series.flatten()),window_size=6)
# 결과 형태(shape)를 확인합니다
Xs.shape, Ys.shape


# In[18]:


Xs[:5]


# 5개의 임의의 index로 추출하여 잘 구성이 되었는지 확인합니다.

# In[19]:


print(Xs[-15:-10])


# In[20]:


print(Ys[-15:-10])


# `np.expand_dims()`
# 
# - 시계열 데이터 생성을 위한 차원 추가 `(number of dataset, window_size, features)`

# In[21]:


# 맨 끝에 차원을 1개 추가 후 shape 확인
np.expand_dims(Xs, -1).shape


# In[22]:


Xs = np.expand_dims(Xs, -1)
Xs.shape, Ys.shape


# ### 4.3 데이터셋 분할
# 
# - 마지막 45개의 데이터는 검증용 데이터셋으로 나머지 데이터를 학습용 데이터셋으로 활용
# - `train`: 학습용 데이터셋
# - `test`: 검증용 데이터셋

# In[23]:


# 데이터셋 분할
x_train, y_train = Xs[:-45], Ys[:-45]
x_test, y_test = Xs[-45:], Ys[-45:]


# In[24]:


# train 데이터셋 shape 확인 (N, W, F), (Target,)
# 417개의 데이터, window_size=6, 하나의 데이터셋 1
# 예측치 417개
x_train.shape, y_train.shape


# In[25]:


# test 데이터셋 shape 확인 (N, W, F), (Target,)
x_test.shape, y_test.shape


# ### 4.4 텐서 데이터셋
# 
# - 딥러닝 모델에 배치 단위로 데이터셋을 주입하기 위하여 TensorDataset 생성

# In[28]:


# 딥러닝 FrameWork
import torch

# device 설정 (cuda 혹은 cpu) gpu가 있을 시, cuda 이용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[29]:


# FloatTensor 변환을 위한 함수
 
def make_tensor(x, device):
    return torch.FloatTensor(x).to(device)


# In[30]:


# FloatTensor 형태로 (train, test) 데이터셋 변환
x_train = make_tensor(x_train, device=device)
y_train = make_tensor(y_train, device=device)
x_test = make_tensor(x_test, device=device)
y_test = make_tensor(y_test, device=device)


# In[31]:


x_train


# `torch.utils.data.TensorDataset` 로 변환하는 이유
# - X, Y 데이터를 묶어서 지도학습용 데이터셋으로 생성
# - 배치(Batch) 구성을 위하여 `DataLoader`로 변환하기 위함

# In[32]:


# TensorDataset 생성
train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)


# 배치 구성을 위한 `torch.utils.data.DataLoader`를 생성합니다.

# In[33]:


# batch_size 설정
BATCH_SIZE = 32

# train_loader, test_loader 생성
train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test, 
                                          batch_size=BATCH_SIZE, 
                                          shuffle=False)


# In[34]:


# 1개 batch 추출
x, y = next(iter(train_loader))


# In[35]:


# x, y 데이터 shape 확인 (batch_size, window_size, features)
x.shape, y.shape


# ## 5. 딥러닝 모델 정의
# ---
# LSTM을 활용한 시계열 예측 모형을 생성합니다.

# ### 5.1 모델 정의에 필요한 모듈 import

# In[36]:


import torch
import torch.nn as nn
import torch.optim as optim


# ### 5.2 LSTM 입출력

# In[37]:


input_size = 1
hidden_size = 32
num_layers = 1

lstm = nn.LSTM(input_size=input_size, 
               hidden_size=hidden_size, 
               num_layers=num_layers, 
               bidirectional=True,
               batch_first=True)


# LSTM **입력 데이터** 정의

# In[38]:


# (batch_size, window_size, input_size)
x = torch.zeros(32, 6, 1)
x.shape


# LSTM 출력

# In[39]:


# LSTM 출력 확인
lstm_out, (h, c) = lstm(x)


# In[40]:


# lstm_out: (batch_size, window_size, hidden_size*bidirectional)
# h       : (bidirectional*num_layers, batch_size, hidden_size)
# c       : (bidirectional*num_layers, batch_size, hidden_size)
lstm_out.shape, h.shape, c.shape


# In[41]:


lstm_out[:,-1,:].shape


# ### 5.3 모델 정의

# #### **[TODO]** 철근 가격 데이터 처리를 위한 LSTM 모델을 정의해주세요.
# 1. LSTM 
# 	- 입력 크기: `input_size`
# 	- 은닉층 크기: `hidden_size`
# 	- 레이어 수: `num_layers`
# 	- 양방향 여부: `bidirectional`
# 	- 모델을 정의할 때, `batch_first=True` 인자를 포함시켜주세요.
# 2. 출력층
# 	- 입력 크기: `hidden_size * self.bidirectional`
# 	- 출력 크기: `output_size`
# 

# In[42]:


# 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional=False):
        super(LSTMModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = 2 if bidirectional else 1
        
        # LSTM 정의
        # 철근 가격 데이터 처리를 위한 LSTM 모델을 정의해주세요
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional,
                            batch_first=True)
        
        # 출력층 정의 (Linear)
        # 철근 가격 예측을 위한 출력층을 정의해주세요
        self.fc = nn.Linear(hidden_size*self.bidirectional, out_features=output_size)
        
    def reset_hidden_state(self, batch_size):
        # hidden state, cell state (bidirectional*num_layers, batch_size, hidden_size)
        self.hidden = (
            # hidden state
            torch.zeros(self.bidirectional*self.num_layers, batch_size, self.hidden_size),
            # cell state
            torch.zeros(self.bidirectional*self.num_layers, batch_size, self.hidden_size)
        )
        
    def forward(self, x):
        # LSTM
        # 코드입력
        output, (h, c) = self.lstm(x, self.hidden)
        # 출력층으로 (맨 끝 Cell의 출력 결과만) 리턴하는 출력층을 정의해주세요
        output = self.fc(output[:,-1,:])
        return output


# In[43]:


# 모델 생성
model = LSTMModel(input_size=1, hidden_size=32, output_size=1, num_layers=1, bidirectional=False)


# In[44]:


# batch_size를 입력 값으로 대입하여 초기화
model.reset_hidden_state(1)


# In[45]:


# batch dimension 추가
inputs = np.expand_dims(x_train[0], 0)
inputs.shape


# In[46]:


# model에 입력 데이터 대입 후 출력 확인
model(torch.FloatTensor(inputs))


# ## 6. 딥러닝 모델 훈련
# ---
# - 손실함수, 옵티마이저 등 모델 훈련에 설정할 하이퍼파라미터를 정의합니다.
# - 모델을 훈련(training) 합니다.

# ### 6.1 하이퍼파라미터 정의
# - 손실함수, 학습율 (Learning rate), 옵티마이저를 정의합니다.
# 
# #### **[TODO]** 손실함수를 정의해서 `loss_fn` 에 할당해주세요.

# In[47]:


# 손실함수 (HuberLoss) 를 정의해주세요
loss_fn = nn.HuberLoss()

# 학습율
lr = 1e-4

# Epoch
num_epochs = 1000

# 옵티마이저
optimizer = optim.Adam(model.parameters(), lr=lr)


# ### 6.2 훈련

# #### **[TODO]** 주석과 영상의 안내에 따라, 모델을 훈련시키는 함수를 완성해주세요.

# In[48]:


training_losses = [] 
validation_losses = []

for epoch in range(num_epochs):
    # 훈련, 검증 손실 
    training_loss = 0.0
    validation_loss = 0.0
    
    # 훈련 모드
    model.train()
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # 옵티마이저 그라디언트 초기화
        # 코드입력
        optimizer.zero_grad()
        
        # 모델 hidden state 초기화
        model.reset_hidden_state(len(x))
        
        # 모델의 추론값을 output 변수에 할당하는 코드를 작성해주세요.
        output = model(x)
        
        # 훈련 데이터에 대한 손실값을 loss 변수에 할당하는 코드를 작성해주세요.
        loss = loss_fn(output, y)
        
        # 손실값의 역전파를 위해 손실 값의 미분값을 계산하는 코드를 작성해주세요.
        loss.backward()
        
        # optimizer를 활용해서 그라디언트 업데이트 (역전파) 를 수행하는 코드를 작성해주세요.
        optimizer.step()
        
        # 훈련 손실 누적
        training_loss += loss.item()*len(x)
    # 평균 훈련 손실
    avg_training_loss = training_loss / len(train_loader)
    training_losses.append(avg_training_loss)

    # 검증 모드
    model.eval()
    
    # 그라디언트 추적 Off
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # 모델 hidden state 초기화
            model.reset_hidden_state(len(x))
            
            # 모델의 추론값을 output 변수에 할당하는 코드를 작성해주세요.
            output = model(x)
            
            # 검증 데이터에 대한 손실값을 loss 변수에 할당하는 코드를 작성해주세요.
            val_loss = loss_fn(output,y)
            
            # 검증 손실 누적
            validation_loss += val_loss.item()*len(x)

    avg_validation_loss = validation_loss / len(test_loader)
    validation_losses.append(avg_validation_loss)
    
    if epoch % 50 == 0:
        print(f'epoch: {epoch+1:03d}, loss: {avg_training_loss:.4f}, val_loss: {avg_validation_loss:.4f}')


# In[52]:


# 학습/검증 손실 시각화
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 4)
ax.plot(training_losses, color='grey', linewidth=1.0, label='Training Loss')
ax.plot(validation_losses, color='red', linewidth=1.0, label='Validation Loss')
ax.set_title('Losses')
ax.legend()
plt.show()


# ## 7. 추론 및 시각화

# ### 7.1 추론

# #### **[TODO]** 주석과 영상의 안내에 따라, 모델을 테스트하는 코드를 완성해주세요.

# In[53]:


preds = []
y_trues = []

# 검증 모드 활성화
model.eval()

# 그라디언트 추적 Off
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        model.reset_hidden_state(len(x))
        # 모델의 추론값을 output 변수에 할당하는 코드를 작성해주세요.
        output = model(x)
        
        # output과 y true inverse transform으로 원래 가격 복원
        pred = scaler.inverse_transform(output.detach().numpy())
        y_true = scaler.inverse_transform(y.detach().numpy().reshape(-1, 1))
        
        # preds, y_trues에 복원된 값 추가
        preds.extend(pred.tolist())
        y_trues.extend(y_true.tolist())


# ### 7.2 추론 결과 시각화

# In[54]:


# 45일 간의 추론 vs 실제 가격 시각화
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
ax.plot(preds, label='Prediction', color='tomato', linestyle='-.')
ax.plot(y_trues, label='Actual', color='skyblue')
ax.legend()
ax.set_title('Prediction on Validation(45 days)')
plt.show()


# ### 7.3 전체 데이터에 대한 추론 및 시각화

# #### **[TODO]** 주석과 영상의 안내에 따라, 학습 데이터에 대한 추론 결과를 시각화 하는 코드를 완성해주세요.

# In[55]:


preds = []
y_trues = []

# 검증 모드 활성화
model.eval()

# 그라디언트 추적 Off
with torch.no_grad():
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        model.reset_hidden_state(len(x))
        
        # 모델의 추론값을 output 변수에 할당하는 코드를 작성해주세요.
        output =model(x)
        
        # output과 y true inverse transform으로 원래 가격 복원
        pred = scaler.inverse_transform(output.detach().numpy())
        y_true = scaler.inverse_transform(y.detach().numpy().reshape(-1, 1))
        
        # preds, y_trues에 복원된 값 추가
        preds.extend(pred.tolist())
        y_trues.extend(y_true.tolist())


# In[56]:


# Training Data 추론 vs 실제 가격 시각화
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
ax.plot(preds, label='Prediction', color='tomato', linestyle='-.')
ax.plot(y_trues, label='Actual', color='skyblue')
ax.legend()
ax.set_title('Prediction on Training Data')
plt.show()


# In[ ]:




