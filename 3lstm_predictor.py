import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import warnings
warnings.filterwarnings('ignore')

# 設定隨機種子確保論文實驗可重複
torch.manual_seed(42)
np.random.seed(42)

ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

def create_sequences(data, seq_length=60):
    """將時間序列資料轉換為 (樣本數, 時間步長, 特徵數)"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# ==========================================
# 🚀 論文 2.2 節要求之 Bi-LSTM + Transformer 混合模型架構
# ==========================================
class BiLSTMTransformerModel(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super(BiLSTMTransformerModel, self).__init__()
        
        # 特徵投影層
        self.feature_mapping = nn.Linear(num_features, d_model)
        
        # 雙向 LSTM (Bi-LSTM) 層
        self.bilstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True # 啟用雙向
        )
        
        self.lstm_proj = nn.Linear(d_model * 2, d_model)
        
        # Transformer Encoder 層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 輸出層
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_features)

    def forward(self, x):
        x = self.feature_mapping(x) 
        lstm_out, _ = self.bilstm(x) 
        lstm_out = self.lstm_proj(lstm_out) 
        transformer_out = self.transformer_encoder(lstm_out) 
        out = transformer_out[:, -1, :] # 取最後一個時間步
        out = self.dropout(out)
        out = self.fc(out) 
        return out

def train_pytorch_model(X_train, y_train, num_features, epochs=30, batch_size=32):
    """PyTorch 模型訓練管線"""
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = BiLSTMTransformerModel(num_features=num_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def clean_returns(df_returns):
    df_returns = df_returns[ASSETS].copy()
    df_returns = df_returns.replace([np.inf, -np.inf], np.nan)
    df_returns = df_returns.fillna(0.0)
    return df_returns

def main():
    print("=== 啟動 Bi-LSTM + Transformer 預期報酬 (μ) 預測與品質驗證引擎 (0602版) ===")
    
    df_returns = pd.read_csv("chip4_usd_returns.csv", index_col="Date", parse_dates=True)
    df_returns = clean_returns(df_returns)
    assets = ASSETS
    
    target_dates = [
        "2019-05-13", "2019-06-19", "2019-06-24", 
        "2020-03-04", "2026-04-27"
    ]
    
    predicted_mu_dict = {}
    performance_records = []
    
    for target_date in target_dates:
        print(f"\n[處理中] 正在針對調倉日 {target_date} 進行模型訓練與檢定...")
        target_timestamp = pd.to_datetime(target_date)
        
        if target_timestamp not in df_returns.index:
            continue
            
        # 嚴格切分：只使用 T-1 之前的資料
        historical_data = df_returns.loc[:target_timestamp].iloc[:-1] 
        
        if len(historical_data) < 65: continue
            
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(historical_data.values)
        
        seq_length = 60
        X_train, y_train = create_sequences(scaled_data, seq_length)
        
        # 訓練混合模型 (論文定稿建議 epochs 可調至 30~50 增加穩定性)
        model = train_pytorch_model(X_train, y_train, num_features=len(assets), epochs=30, batch_size=32)
        
        model.eval()
        last_60_days = scaled_data[-seq_length:]
        last_60_days_tensor = torch.tensor(last_60_days, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            scaled_prediction = model(last_60_days_tensor).numpy()
        
        model_prediction = scaler.inverse_transform(scaled_prediction)[0]
        
        # ==========================================
        # 🚨 論文 4.1 節：品質驗證與對照組計算
        # ==========================================
        # 建立 Naïve Baseline (歷史 60 天平均) 作為對照組
        naive_prediction = historical_data.iloc[-seq_length:].mean().values
        
        # 抓取目標日當天「真實發生」的報酬率
        actual_real_return = df_returns.loc[target_timestamp].values
        
        # 計算誤差指標 RMSE / MAE
        model_rmse = math.sqrt(mean_squared_error(actual_real_return, model_prediction))
        model_mae = mean_absolute_error(actual_real_return, model_prediction)
        naive_rmse = math.sqrt(mean_squared_error(actual_real_return, naive_prediction))
        naive_mae = mean_absolute_error(actual_real_return, naive_prediction)
        
        performance_records.append({
            "Market_Scenario": target_date,
            "Model_RMSE": model_rmse, "Model_MAE": model_mae,
            "Naive_RMSE": naive_rmse, "Naive_MAE": naive_mae
        })
        
        mu_vector = {assets[i]: float(model_prediction[i]) for i in range(len(assets))}
        predicted_mu_dict[target_date] = mu_vector
        print(f"  ✅ {target_date} 檢定完成！(Model RMSE: {model_rmse:.5f} | Naive RMSE: {naive_rmse:.5f})")

    # 輸出供 QUBO 讀取的預期報酬
    output_json = "lstm_predicted_mu.json"
    with open(output_json, 'w') as f:
        json.dump(predicted_mu_dict, f, indent=4, allow_nan=False)
        
    # 印出論文 4.1 節專用表格
    df_perf = pd.DataFrame(performance_records)
    print("\n" + "="*75)
    print("📊 論文 4.1 節：BiLSTM-Transformer 預測品質驗證明細表")
    print("="*75)
    print(df_perf.to_string(index=False, formatters={
        'Model_RMSE': '{:,.6f}'.format, 'Model_MAE': '{:,.6f}'.format,
        'Naive_RMSE': '{:,.6f}'.format, 'Naive_MAE': '{:,.6f}'.format
    }))
    print("="*75)

if __name__ == "__main__":
    main()