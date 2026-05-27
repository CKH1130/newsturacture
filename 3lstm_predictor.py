import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
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
# 🚀 1. 新增：Bi-LSTM + Transformer 混合模型
# ==========================================
class BiLSTMTransformerModel(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super(BiLSTMTransformerModel, self).__init__()
        
        # 特徵投影層：將原始特徵數對齊到 Transformer 的 d_model 维度
        self.feature_mapping = nn.Linear(num_features, d_model)
        
        # 雙向 LSTM (Bi-LSTM) 層：捕捉局部雙向時序特徵
        # bidirectional=True 會讓 output 维度變成 d_model * 2
        self.bilstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 將 Bi-LSTM 的雙向輸出 (d_model * 2) 重新投影回 d_model
        self.lstm_proj = nn.Linear(d_model * 2, d_model)
        
        # Transformer Encoder 層：捕捉長距離全局自注意力特徵
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 輸出層：預測 15 檔股票的下一步報酬率
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_features)

    def forward(self, x):
        # x 形狀: [batch_size, seq_length, num_features]
        x = self.feature_mapping(x) # -> [batch_size, seq_length, d_model]
        
        # 經過 Bi-LSTM
        lstm_out, _ = self.bilstm(x) # -> [batch_size, seq_length, d_model * 2]
        lstm_out = self.lstm_proj(lstm_out) # -> [batch_size, seq_length, d_model]
        
        # 經過 Transformer Encoder
        transformer_out = self.transformer_encoder(lstm_out) # -> [batch_size, seq_length, d_model]
        
        # 取最後一個時間步 (Last Time Step) 的輸出進行預測
        out = transformer_out[:, -1, :] # -> [batch_size, d_model]
        out = self.dropout(out)
        out = self.fc(out) # -> [batch_size, num_features]
        return out

def train_pytorch_model(X_train, y_train, num_features, epochs=20, batch_size=32):
    """PyTorch 模型訓練管線"""
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = BiLSTMTransformerModel(num_features=num_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練循環
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
    missing_before = int(df_returns.isna().sum().sum())
    if missing_before:
        print(f"[Data Check] 偵測到 {missing_before} 個缺失/無限值，已以 0.0 補值。")
    df_returns = df_returns.fillna(0.0)
    return df_returns

def main():
    print("=== 啟動 Bi-LSTM + Transformer 預期報酬 (μ) 預測引擎 ===")
    
    # 1. 讀取真實歷史報酬率
    df_returns = pd.read_csv("chip4_usd_returns.csv", index_col="Date", parse_dates=True)
    df_returns = clean_returns(df_returns)
    assets = ASSETS
    
    # 2. 5 大代表日
    target_dates = [
        "2019-05-13", # 極端大跌日
        "2019-06-19", # 極端大漲日
        "2019-06-24", # 平穩日
        "2020-03-04", # 高波動日
        "2026-04-08"  # 事件衝擊日
    ]
    
    predicted_mu_dict = {}
    
    for target_date in target_dates:
        print(f"\n[處理中] 正在針對調倉日 {target_date} 進行 Bi-LSTM + Transformer 預測...")
        
        target_timestamp = pd.to_datetime(target_date)
        historical_data = df_returns.loc[:target_timestamp].iloc[:-1] 
        
        if len(historical_data) < 65:
            print(f"  -> 警告: {target_date} 前的歷史數據過少，跳過。")
            continue
            
        # 3. 資料正規化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(historical_data.values)
        if not np.isfinite(scaled_data).all():
            print(f"  -> 警告: {target_date} 的訓練資料仍含非有限值，跳過。")
            continue
        
        # 4. 製作特徵序列
        seq_length = 60
        X_train, y_train = create_sequences(scaled_data, seq_length)
        
        # 5. 訓練新模型
        # 論文測試：Epoch 可設 20，跑定稿實驗時建議調至 50-100
        print(f"  -> 開始訓練 Bi-LSTM + Transformer 模型 (資料筆數: {len(X_train)})...")
        model = train_pytorch_model(X_train, y_train, num_features=len(assets), epochs=20, batch_size=32)
        
        # 6. 預測目標日 T 的報酬率
        model.eval()
        last_60_days = scaled_data[-seq_length:]
        last_60_days_tensor = torch.tensor(last_60_days, dtype=torch.float32).unsqueeze(0) # 增加 batch 维度
        
        with torch.no_grad():
            scaled_prediction = model(last_60_days_tensor).numpy()
        
        # 反轉回真實的報酬率百分比
        actual_prediction = scaler.inverse_transform(scaled_prediction)[0]
        if not np.isfinite(actual_prediction).all():
            print(f"  -> 警告: {target_date} 的模型預測含非有限值，跳過。")
            continue
        
        # 存入字典
        mu_vector = {assets[i]: float(actual_prediction[i]) for i in range(len(assets))}
        predicted_mu_dict[target_date] = mu_vector
        
        print(f"  ✅ {target_date} 預測完成！(例如 NVDA 預期報酬: {mu_vector['NVDA']*100:.2f}%)")

    # 7. 儲存成 JSON 檔供 QUBO 模型使用
    output_file = "lstm_predicted_mu.json"
    with open(output_file, 'w') as f:
        json.dump(predicted_mu_dict, f, indent=4, allow_nan=False)
        
    print(f"\n🎉 成功！所有代表日的預期報酬 (μ) 已更新為新混合架構預測，並匯出至 '{output_file}'")

if __name__ == "__main__":
    main()