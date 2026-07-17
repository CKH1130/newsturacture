import json
import numpy as np
import itertools

ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def build_inputs_for_date(date, mu_data, sigma_data, assets):
    missing_mu = [ticker for ticker in assets if ticker not in mu_data[date]]
    if missing_mu:
        raise ValueError(f"μ 缺少 ticker: {missing_mu}")

    try:
        mu = np.array([mu_data[date][ticker] for ticker in assets], dtype=float)
        sigma = np.array(sigma_data[date], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"μ/Σ 含無法轉成數值的資料: {exc}") from exc

    if sigma.shape != (len(assets), len(assets)):
        raise ValueError(f"Σ 矩陣形狀錯誤: {sigma.shape}")

    if not np.isfinite(mu).all():
        bad_tickers = [assets[i] for i, value in enumerate(mu) if not np.isfinite(value)]
        raise ValueError(f"μ 含非有限值: {bad_tickers}")

    if not np.isfinite(sigma).all():
        bad_count = int((~np.isfinite(sigma)).sum())
        raise ValueError(f"Σ 含 {bad_count} 個非有限值")

    return mu, sigma

def main():
    print("=== 啟動古典暴力破解引擎 (Brute Force Exact Solver) ===\n")
    
    try:
        mu_data = load_json_data("lstm_predicted_mu.json")
        sigma_data = load_json_data("sigma_matrices.json")
    except FileNotFoundError:
        print("❌ 找不到 JSON 檔案，請確認 mu 和 sigma 的萃取程式是否都已執行完畢。")
        return
        
    assets = ASSETS
    
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
    
    target_dates = [
        "2019-05-13", "2019-06-19", "2019-06-24", 
        "2020-03-04", "2026-04-27"
    ]
    
    print(f"👉 參數設定: λ = {lmbda}, P3 = {p3}")
    print("👉 開始窮舉 C(15, 4) = 1365 種組合...\n")
    
    for date in target_dates:
        if date not in mu_data or date not in sigma_data:
            print(f"跳過 {date}，缺乏數據。")
            continue
            
        try:
            mu, sigma = build_inputs_for_date(date, mu_data, sigma_data, assets)
        except ValueError as exc:
            print("==================================================")
            print(f"📅 調倉日: {date}")
            print(f"❌ 跳過：輸入資料無效，{exc}")
            print("==================================================\n")
            continue
        
        best_energy = float('inf')
        best_portfolio = None
        best_details = {}
        feasible_count = 0
        
        # 產生所有 15 選 4 的組合 (C(15,4) = 1365)
        all_combinations = list(itertools.combinations(range(15), 4))
        
        for combo in all_combinations:
            x = np.zeros(15)
            x[list(combo)] = 1
            
            selected_tickers = [assets[i] for i in combo]
            
            # --- 檢查硬限制 (P2: 市場配置 1,1,1,1) ---
            us_count = sum(1 for t in selected_tickers if t in market_mapping["US"])
            tw_count = sum(1 for t in selected_tickers if t in market_mapping["TW"])
            jp_count = sum(1 for t in selected_tickers if t in market_mapping["JP"])
            kr_count = sum(1 for t in selected_tickers if t in market_mapping["KR"])
            
            # 市場配額改為 1,1,1,1 的絕對對稱
            if not (tw_count == 1 and us_count == 1 and jp_count == 1 and kr_count == 1):
                continue 
            feasible_count += 1
                
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
            
            total_energy = h_rr + h_dep
            if not np.isfinite(total_energy):
                continue
            
            if total_energy < best_energy:
                best_energy = total_energy
                best_portfolio = selected_tickers
                best_details = {
                    "Risk": risk,
                    "Return": return_val,
                    "Violations": penalty_violations
                }
                
        print("==================================================")
        print(f"📅 調倉日: {date}")
        if best_portfolio is None:
            print("❌ 找不到可行且有限的投資組合。")
            print(f"   通過市場配置的組合數: {feasible_count}")
            print("==================================================\n")
            continue

        print(f"🏆 絕對最佳解 (Energy: {best_energy:.6f})")
        print(f"💼 投資組合: {best_portfolio}")
        print(f"📊 預期投組報酬(無加權): {best_details['Return']*100:.2f}% | 投組變異數: {best_details['Risk']:.6f}")
        print(f"⚠️ 供應鏈違規次數: {best_details['Violations']}")
        print("==================================================\n")

if __name__ == "__main__":
    main()