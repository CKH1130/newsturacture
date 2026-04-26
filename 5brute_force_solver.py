import json
import numpy as np
import itertools
import pandas as pd

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    print("=== 啟動古典暴力破解引擎 (Brute Force Exact Solver) ===\n")
    
    # 1. 讀取我們辛苦萃取出來的真實數據
    try:
        mu_data = load_json_data("lstm_predicted_mu.json")
        sigma_data = load_json_data("sigma_matrices.json")
    except FileNotFoundError:
        print("❌ 找不到 JSON 檔案，請確認 mu 和 sigma 的萃取程式是否都已執行完畢。")
        return
        
    assets = [
        "NVDA", "AMD", "QCOM", "AMAT", "ASML",
        "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
        "8035.T", "6857.T", "4063.T",
        "005930.KS", "000660.KS", "042700.KS"
    ]
    
    # 定義市場對應關係
    market_mapping = {
        "US": ["NVDA", "AMD", "QCOM", "AMAT", "ASML"],
        "TW": ["2330.TW", "2454.TW", "3711.TW", "6488.TWO"],
        "JP": ["8035.T", "6857.T", "4063.T"],
        "KR": ["005930.KS", "000660.KS", "042700.KS"]
    }
    
    # 定義供應鏈依賴
    dependencies = [
        ("NVDA", "2330.TW"),
        ("AMD", "2330.TW"),
        ("2330.TW", "ASML")
    ]
    
    # 論文鎖定的參數
    lmbda = 0.5
    p3 = 10.0
    
    # 我們要測試的 5 個代表日
    target_dates = [
        "2019-05-13", "2019-06-19", "2019-06-24", 
        "2020-03-04", "2026-04-08"
    ]
    
    print(f"👉 參數設定: λ = {lmbda}, P3 = {p3}")
    print("👉 開始窮舉 C(15, 5) = 3003 種組合...\n")
    
    # 針對每一個代表日進行暴力破解
    for date in target_dates:
        if date not in mu_data or date not in sigma_data:
            print(f"跳過 {date}，缺乏數據。")
            continue
            
        # 將 JSON 的數據轉回 Numpy 格式
        mu = np.array([mu_data[date][ticker] for ticker in assets])
        sigma = np.array(sigma_data[date])
        
        best_energy = float('inf')
        best_portfolio = None
        best_details = {}
        
        # 產生所有 15 選 5 的組合 (回傳的是 index 的 tuple)
        all_combinations = list(itertools.combinations(range(15), 5))
        
        for combo in all_combinations:
            # 建立二元向量 x (選中為1，未選為0)
            x = np.zeros(15)
            x[list(combo)] = 1
            
            selected_tickers = [assets[i] for i in combo]
            
            # --- 檢查硬限制 (P2: 市場配置 2,1,1,1) ---
            us_count = sum(1 for t in selected_tickers if t in market_mapping["US"])
            tw_count = sum(1 for t in selected_tickers if t in market_mapping["TW"])
            jp_count = sum(1 for t in selected_tickers if t in market_mapping["JP"])
            kr_count = sum(1 for t in selected_tickers if t in market_mapping["KR"])
            
            if not (tw_count == 2 and us_count == 1 and jp_count == 1 and kr_count == 1):
                continue # 不符合市場配置，直接淘汰 (等同於 P2 給了無限大的懲罰)
                
            # --- 計算目標函數 (Energy) ---
            # 1. H_rr (風險與報酬)
            risk = np.dot(x.T, np.dot(sigma, x))
            return_val = np.dot(mu.T, x)
            h_rr = lmbda * risk - (1 - lmbda) * return_val
            
            # 2. H_dep (供應鏈懲罰 P3)
            penalty_violations = 0
            for dep, relies_on in dependencies:
                if (dep in selected_tickers) and (relies_on not in selected_tickers):
                    penalty_violations += 1
            
            h_dep = p3 * penalty_violations
            
            # 總能量 (越低越好)
            total_energy = h_rr + h_dep
            
            # 更新最佳解
            if total_energy < best_energy:
                best_energy = total_energy
                best_portfolio = selected_tickers
                best_details = {
                    "Risk": risk,
                    "Return": return_val,
                    "Violations": penalty_violations
                }
                
        # 印出該日期的最佳解答
        print("==================================================")
        print(f"📅 調倉日: {date}")
        print(f"🏆 絕對最佳解 (Energy: {best_energy:.6f})")
        print(f"💼 投資組合: {best_portfolio}")
        print(f"📊 預期投組報酬(無加權): {best_details['Return']*100:.2f}% | 投組變異數: {best_details['Risk']:.6f}")
        print(f"⚠️ 供應鏈違規次數: {best_details['Violations']}")
        print("==================================================\n")

if __name__ == "__main__":
    main()
