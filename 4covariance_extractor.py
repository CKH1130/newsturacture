import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

def clean_returns(df_returns):
    df_returns = df_returns[ASSETS].copy()
    df_returns = df_returns.replace([np.inf, -np.inf], np.nan)
    missing_before = int(df_returns.isna().sum().sum())
    if missing_before:
        print(f"[Data Check] 偵測到 {missing_before} 個缺失/無限值，已以 0.0 補值。")
    return df_returns.fillna(0.0)

def main():
    print("=== 啟動 60 日滾動共變異數 (Σ) 萃取引擎 ===\n")
    
    # 1. 讀取真實歷史報酬率
    print("[Step 1] 讀取 chip4_usd_returns.csv ...")
    df_returns = pd.read_csv("chip4_usd_returns.csv", index_col="Date", parse_dates=True)
    
    # 🚨 學術防禦：嚴格鎖定這 15 檔股票的「絕對順序」
    # 這非常重要！如果矩陣的欄位順序亂掉，量子演算法選出來的股票就會對應錯誤
    assets = ASSETS
    
    # 強制將 DataFrame 的欄位排序對齊我們的標準名單
    df_returns = clean_returns(df_returns)
    
    # 2. 我們選定的 5 大代表日
    target_dates = [
        "2019-05-13", # 極端大跌日
        "2019-06-19", # 極端大漲日
        "2019-06-24", # 平穩日
        "2020-03-04", # 高波動日
        "2026-04-27"  # 事件衝擊日
    ]
    
    sigma_dict = {}
    
    # 3. 開始針對每一天萃取矩陣
    print("[Step 2] 開始計算各代表日之共變異數矩陣 (T-60 到 T-1)...\n")
    for date_str in target_dates:
        target_date = pd.to_datetime(date_str)
        
        # 找出該日期在資料表中的索引位置 (Row Index)
        try:
            t_idx = df_returns.index.get_loc(target_date)
        except KeyError:
            print(f"  ❌ 警告：找不到日期 {date_str}，請確認資料庫區間。")
            continue
            
        # 檢查前面是否有足夠的 60 天資料
        if t_idx < 60:
            print(f"  ❌ 警告：日期 {date_str} 前的歷史資料不足 60 天，無法計算。")
            continue
            
        # 🚨 核心邏輯：精準切出 T-60 到 T-1 的資料 (絕對不能包含 T 當天！)
        historical_60_days = df_returns.iloc[t_idx - 60 : t_idx]
        
        # 計算共變異數矩陣 (Covariance Matrix)
        cov_matrix = historical_60_days.cov()
        if not np.isfinite(cov_matrix.values).all():
            print(f"  ❌ 警告：日期 {date_str} 的共變異數矩陣含非有限值，跳過。")
            continue
        
        # 將 DataFrame 轉為二維陣列 (List of Lists)，以便存入 JSON
        sigma_dict[date_str] = cov_matrix.values.tolist()
        
        print(f"  ✅ 成功萃取 {date_str} 矩陣 (形狀: {cov_matrix.shape})")

    # 4. 存檔輸出
    output_file = "sigma_matrices.json"
    with open(output_file, "w") as f:
        json.dump(sigma_dict, f, indent=4, allow_nan=False)
        
    print(f"\n🎉 大功告成！5 個代表日的共變異數矩陣 (Σ) 已成功儲存至 '{output_file}'")

if __name__ == "__main__":
    main()
