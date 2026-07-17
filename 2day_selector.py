import pandas as pd
import numpy as np
import yfinance as yf

def get_adjusted_close(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"無法下載 {ticker} 資料，請確認網路連線或 Yahoo Finance 資料狀態。")

    if isinstance(data.columns, pd.MultiIndex):
        price_field = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
        return data.xs(price_field, level=0, axis=1).iloc[:, 0]

    price_field = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    return data[price_field]

def main():
    print("=== 啟動 5 大代表日客觀篩選器 (含 CAPM 異常報酬檢定) ===\n")
    
    # 1. 讀取我們先前整理好的 15 檔股票報酬率
    df_returns = pd.read_csv("chip4_usd_returns.csv", index_col="Date", parse_dates=True)
    
    # 計算 15 檔股票的「平均單日報酬率」(等權重投資組合) 作為市場狀態指標
    portfolio_return = df_returns.mean(axis=1)
    
    # 計算 5 日滾動波動率 (標準差)
    rolling_vol = portfolio_return.rolling(window=5).std()
    
    # 2. CAPM 模型準備：下載 SOXX (費城半導體 ETF) 作為市場基準
    print("[Step 1] 下載 SOXX 基準指數計算 CAPM...")
    soxx = get_adjusted_close("SOXX", start=df_returns.index[0], end=df_returns.index[-1])
    market_return = soxx.pct_change().dropna()
    
    # 對齊日期
    aligned_data = pd.concat([portfolio_return, market_return], axis=1, join='inner').dropna()
    aligned_data.columns = ['Portfolio', 'Market']
    
    # 3. 計算 CAPM 異常報酬 (Abnormal Return, AR)
    print("[Step 2] 計算 Abnormal Return (AR) 與 2σ 門檻...")
    # 簡化版 CAPM: 假設無風險利率 Rf = 0，Beta = Cov(P, M) / Var(M)
    cov_matrix = np.cov(aligned_data['Portfolio'], aligned_data['Market'])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # 預期報酬 = Beta * Market Return
    expected_return = beta * aligned_data['Market']
    # 異常報酬 (AR) = 實際報酬 - 預期報酬
    ar = aligned_data['Portfolio'] - expected_return
    
    # 計算 AR 的 2 個標準差 (2 Sigma)
    ar_std_2 = 2 * ar.std()
    print(f"  -> 投組 Beta 值: {beta:.2f}")
    print(f"  -> AR 2σ 門檻值: ±{ar_std_2*100:.2f}%")
    
    # 4. 定義五大情境的日期集合
    print("\n[Step 3] 依照分位數與統計特徵篩選日期...")
    # (1) 事件日：|AR| > 2σ
    event_days = ar[abs(ar) > ar_std_2].index
    
    # (2) 大跌日：報酬率最低 5%
    q05_ret = portfolio_return.quantile(0.05)
    crash_days = portfolio_return[portfolio_return <= q05_ret].index
    
    # (3) 大漲日：報酬率最高 5%
    q95_ret = portfolio_return.quantile(0.95)
    boom_days = portfolio_return[portfolio_return >= q95_ret].index
    
    # (4) 高波動日：波動率最高 5%
    q95_vol = rolling_vol.quantile(0.95)
    high_vol_days = rolling_vol[rolling_vol >= q95_vol].index
    
    # (5) 平穩日：報酬率與波動率都在 45%~55% 之間
    normal_ret_days = portfolio_return[(portfolio_return >= portfolio_return.quantile(0.45)) & 
                                       (portfolio_return <= portfolio_return.quantile(0.55))].index
    normal_vol_days = rolling_vol[(rolling_vol >= rolling_vol.quantile(0.45)) & 
                                  (rolling_vol <= rolling_vol.quantile(0.55))].index
    normal_days = normal_ret_days.intersection(normal_vol_days)
    
    # 5. 執行「排他與優先規則」挑選最終 5 天
    print("\n[Step 4] 執行排他規則，選定最終 5 大代表日...")
    selected_dates = {}
    used_dates = set()

    def select_date(candidate_days, category_name):
        for d in candidate_days:
            # 確保有過去 60 天資料算共變異數
            if d not in used_dates and df_returns.index.get_loc(d) >= 60:
                selected_dates[category_name] = d.strftime('%Y-%m-%d')
                used_dates.add(d)
                return d
        return None

    # 嚴格依照優先順序挑選：事件 -> 大跌 -> 大漲 -> 波動 -> 平穩
    evt_d = select_date(event_days.sort_values(ascending=False), "事件衝擊日 (AR > 2σ)")
    cra_d = select_date(crash_days, "極端大跌日 (Bottom 5%)")
    boo_d = select_date(boom_days, "極端大漲日 (Top 5%)")
    vol_d = select_date(high_vol_days, "高波動日 (Vol Top 5%)")
    nor_d = select_date(normal_days, "平穩日 (45%-55%)")
    
    # 6. 印出最終結果
    print("\n=========================================")
    print("🎯 最終選定之 5 大代表調倉日 (OOS 起點)")
    print("=========================================")
    for category, date in selected_dates.items():
        if category.startswith("事件"):
            # 找出該事件日的 AR 值
            actual = aligned_data.loc[date, 'Portfolio']
            abnormal = ar.loc[date]
            print(f"✅ {category}: {date} (實際報酬: {actual*100:.2f}%, 異常報酬: {abnormal*100:.2f}%)")
        else:
            print(f"✅ {category}: {date}")
    
    print("\n(註：此 5 個日期將寫死於論文中，作為後續所有演算法 (Brute Force, QAOA) 的共同測試基準。)")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
