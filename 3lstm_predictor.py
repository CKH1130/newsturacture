import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

def create_sequences(data, seq_length=60):
    """
    將時間序列資料轉換為 LSTM 需要的 3D 結構: (樣本數, 時間步長, 特徵數)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length, num_features):
    """
    建立標準 LSTM 模型 (論文中的預測引擎)
    """
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(seq_length, num_features)),
        Dropout(0.2), # 防止過擬合
        Dense(num_features) # 輸出層: 15 檔股票的預期報酬
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def main():
    print("=== 啟動 LSTM 預期報酬 (μ) 預測引擎 ===")
    
    # 1. 讀取真實歷史報酬率
    df_returns = pd.read_csv("chip4_usd_returns.csv", index_col="Date", parse_dates=True)
    assets = df_returns.columns.tolist()
    
    # 2. 我們在上一步嚴格篩選出的 5 大代表日
    target_dates = [
        "2019-05-13", # 極端大跌日
        "2019-06-19", # 極端大漲日
        "2019-06-24", # 平穩日
        "2020-03-04", # 高波動日
        "2026-04-08"  # 事件衝擊日
    ]
    
    # 儲存預測結果的字典
    predicted_mu_dict = {}
    
    # 針對每一個代表日，獨立訓練模型並預測 (防止資料洩漏)
    for target_date in target_dates:
        print(f"\n[處理中] 正在針對調倉日 {target_date} 進行 LSTM 預測...")
        
        target_timestamp = pd.to_datetime(target_date)
        
        # 🚨 學術防禦：切出 T-1 之前的資料 (絕對不偷看未來)
        historical_data = df_returns.loc[:target_timestamp].iloc[:-1] 
        
        if len(historical_data) < 65:
            print(f"  -> 警告: {target_date} 前的歷史數據過少，跳過。")
            continue
            
        # 3. 資料正規化 (LSTM 對數值範圍敏感)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(historical_data.values)
        
        # 4. 製作特徵序列 (過去 60 天預測明天)
        seq_length = 60
        X_train, y_train = create_sequences(scaled_data, seq_length)
        
        # 5. 建立並訓練 LSTM 模型
        # 注意：為了節省您測試的時間，這裡 epochs 設為 10。寫論文定稿時可以調高到 50 或 100
        model = build_lstm_model(seq_length, num_features=len(assets))
        print(f"  -> 開始訓練 LSTM 模型 (資料筆數: {len(X_train)})...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # 6. 預測目標日 T 的報酬率
        # 抓取 T-1 往前推 60 天的資料作為輸入
        last_60_days = scaled_data[-seq_length:]
        last_60_days_reshaped = last_60_days.reshape(1, seq_length, len(assets))
        
        scaled_prediction = model.predict(last_60_days_reshaped, verbose=0)
        
        # 將預測結果反轉回真實的報酬率百分比
        actual_prediction = scaler.inverse_transform(scaled_prediction)[0]
        
        # 存入字典
        mu_vector = {assets[i]: float(actual_prediction[i]) for i in range(len(assets))}
        predicted_mu_dict[target_date] = mu_vector
        
        print(f"  ✅ {target_date} 預測完成！(例如 NVDA 預期報酬: {mu_vector['NVDA']*100:.2f}%)")

    # 7. 儲存成 JSON 檔供 QUBO 模型使用
    output_file = "lstm_predicted_mu.json"
    with open(output_file, 'w') as f:
        json.dump(predicted_mu_dict, f, indent=4)
        
    print(f"\n🎉 恭喜！所有代表日的預期報酬 (μ) 已成功匯出至 '{output_file}'")

if __name__ == "__main__":
    main()
