import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


class DataSets():
    def __init__(self, df, target_column, window_size=6, split_days=45, batch_size=32, device='cpu'):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        
        # preprocessing
        series = self.preprocessing(df, target_column)
        
        # scaling
        scaled_series = self.scaling(series)
        
        # dataset
        Xs, Ys = self.make_dataset(pd.Series(scaled_series.flatten()), window_size=window_size)        
        
        # split
        (x_train, y_train), (x_test, y_test) = self.split_dataset(Xs, Ys, split_days)

        # tensor dataset
        x_train = self.make_tensor(x_train, device=device)
        y_train = self.make_tensor(y_train, device=device)
        x_test = self.make_tensor(x_test, device=device)
        y_test = self.make_tensor(y_test, device=device)
        
        train = torch.utils.data.TensorDataset(x_train, y_train)
        test = torch.utils.data.TensorDataset(x_test, y_test)
        
        # data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=train, 
                                                        batch_size=batch_size, 
                                                        shuffle=False)

        self.test_loader = torch.utils.data.DataLoader(dataset=test, 
                                                       batch_size=batch_size, 
                                                       shuffle=False)
        
        
    def preprocessing(self, df, target_column):
        df = df.set_index('Date')
        target_index = pd.date_range('20130107', '20211227', freq='W-MON')
        df = df.loc[df.index.isin(target_index), target_column]
        return df
        
    def scaling(self, data):
        output = self.scaler.fit_transform(data.values.reshape(-1, 1))
        return output
    
    def make_dataset(self, series, window_size=6):
        Xs = []
        Ys = []
        for i in range(len(series) - window_size):
            Xs.append(series.iloc[i:i+window_size].values)
            Ys.append(series.iloc[i+window_size])
        return np.expand_dims(np.array(Xs), -1), np.array(Ys)
    
    def split_dataset(self, Xs, Ys, split_days=45):
        x_train, y_train = Xs[:-split_days], Ys[:-split_days]
        x_test, y_test = Xs[-split_days:], Ys[-split_days:]
        return (x_train, y_train), (x_test, y_test)

    def make_tensor(self, x, device):
        return torch.FloatTensor(x).to(device)
    
    def get_scaler(self):
        return self.scaler
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader



# 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional=False):
        super(LSTMModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = 2 if bidirectional else 1
        
        # LSTM 정의
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional,
                            batch_first=True)
        
        # 출력층 정의
        self.fc = nn.Linear(hidden_size*self.bidirectional, output_size)
        
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
        output, (h, c) = self.lstm(x, self.hidden)
        # 출력층
        output = self.fc(output[:,-1])
        return output
    
    
def train(data_loader, device='cpu', lr=1e-4, num_epochs=2000, seq_length=6, model=None, print_every=100):
    # 손실함수 정의
    loss_fn = nn.HuberLoss()
    
    if model is None:
        # 모델 생성
        model = LSTMModel(input_size=1, hidden_size=32, output_size=1, num_layers=1, bidirectional=False)
    
    # 옵티마이저 정의
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = [] 

    # 훈련모드 
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            # 옵티마이저 그라디언트 초기화
            optimizer.zero_grad()
            # 모델 hidden state 초기화
            model.reset_hidden_state(len(x))
            # 추론
            output = model(x)   
            # 손실 계산
            loss = loss_fn(output, y) 
            # 미분 계산
            loss.backward() 
            # 그라디언트 업데이트
            optimizer.step() 
            # 훈련 손실 계산
            running_loss += loss.item()*len(x)
        # 평균 훈련 손실
        avg_train_loss = running_loss / len(data_loader)
        losses.append(avg_train_loss)
        if epoch % print_every == 0:
            print(f'epoch: {epoch+1}, loss: {avg_train_loss:.4f}')
    
    # 손실과 model 반환
    return losses, model

def evaluate(data_loader, model, scaler, device='cpu'):
    preds = []
    y_trues = []

    # 검증모드
    model.eval()

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            model.reset_hidden_state(len(x))
            # 추론
            output = model(x)
            # 예측값, Grount Truth inverse transform
            pred = scaler.inverse_transform(output.detach().numpy())
            y_true = scaler.inverse_transform(y.detach().numpy().reshape(-1, 1))
            
            preds.extend(pred.tolist())
            y_trues.extend(y_true.tolist())
    
    # Ground Truth와 예측값 반환
    return y_trues, preds