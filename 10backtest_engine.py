import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === 設定繪圖與編碼 ===
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OOS_DAYS = 20 
RISK_FREE_RATE = 0.02 

# 定義四種求解器來源及其繪圖樣式
SOLVERS = {
    "Brute Force": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 2.5},
    "QAOA (Ideal)": {"color": "#2ca02c", "marker": "^", "linestyle": ":", "linewidth": 2.0},
    "QAOA (Noisy)": {"color": "#ff7f0e", "marker": "x", "linestyle": "-.", "linewidth": 2.0},
    "QAOA (IBM Real)": {"color": "#d62728", "marker": "s", "linestyle": "--", "linewidth": 2.5}
}

# === 根據 5, 6, 9 號腳本真實執行結果填入之選股組合 ===
portfolios_data = {
    "2019-05-13": {
        "Brute Force": ["ASML", "6488.TWO", "8035.T", "000660.KS"],
        "QAOA (Ideal)": ["QCOM", "3711.TW", "6857.T", "000660.KS"],
        "QAOA (Noisy)": ["QCOM", "6488.TWO", "6857.T", "042700.KS"],
        "QAOA (IBM Real)": ["QCOM","2330.TW","2454.TW","6857.T","042700.KS"],
    },
    "2019-06-19": {
        "Brute Force": ["AMAT", "6488.TWO", "4063.T", "042700.KS"],
        "QAOA (Ideal)": ["ASML", "6488.TWO", "4063.T", "042700.KS"],
        "QAOA (Noisy)": ["ASML", "2330.TW", "4063.T", "005930.KS"],
        "QAOA (IBM Real)": ["ASML","2330.TW","3711.TW","4063.T","005930.KS","000660.KS"],
    },
    "2019-06-24": {
        "Brute Force": ["ASML", "6488.TWO", "4063.T", "042700.KS"],
        "QAOA (Ideal)": ["QCOM", "3711.TW", "8035.T", "042700.KS"],
        "QAOA (Noisy)": ["ASML", "2330.TW", "4063.T", "000660.KS"],
        "QAOA (IBM Real)": ["AMD", "QCOM", "2330.TW", "8035.T", "000660.KS"],
    },
    "2020-03-04": {
        "Brute Force": ["ASML", "2454.TW", "6857.T", "042700.KS"],
        "QAOA (Ideal)": ["AMAT", "3711.TW", "6857.T", "000660.KS"],
        "QAOA (Noisy)": ["ASML", "2330.TW", "6857.T", "042700.KS"],
        "QAOA (IBM Real)": ["AMD", "AMAT", "4063.T"],
    },
    "2026-04-27": {
        "Brute Force": ["ASML", "3711.TW", "8035.T", "000660.KS"],
        "QAOA (Ideal)": ["ASML", "6488.TWO", "8035.T", "000660.KS"],
        "QAOA (Noisy)": ["QCOM", "3711.TW", "4063.T", "000660.KS"],
        "QAOA (IBM Real)": ["QCOM", "AMAT", "8035.T", "6857.T", "042700.KS"],
    },
}

# === 各求解器於各調倉日之最新真實運算耗時數據 (單位: 秒) ===
execution_times = {
    "2019-05-13": {
        "Brute Force": "0.006",
        "QAOA (Ideal)": "2.21",
        "QAOA (Noisy)": "231.40",
        "QAOA (IBM Real)": "31.97",
    },
    "2019-06-19": {
        "Brute Force": "0.009",
        "QAOA (Ideal)": "1.67",
        "QAOA (Noisy)": "217.93",
        "QAOA (IBM Real)": "32.41",
    },
    "2019-06-24": {
        "Brute Force": "0.009",
        "QAOA (Ideal)": "1.70",
        "QAOA (Noisy)": "219.63",
        "QAOA (IBM Real)": "32.17",
    },
    "2020-03-04": {
        "Brute Force": "0.007",
        "QAOA (Ideal)": "1.75",
        "QAOA (Noisy)": "255.99",
        "QAOA (IBM Real)": "32.00",
    },
    "2026-04-27": {
        "Brute Force": "0.007",
        "QAOA (Ideal)": "1.82",
        "QAOA (Noisy)": "226.18",
        "QAOA (IBM Real)": "32.29",
    },
}

def calculate_metrics(daily_returns, name):
    cum_return = (1 + daily_returns).prod() - 1
    ann_return = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol != 0 else 0
    cum_wealth = (1 + daily_returns).cumprod()
    mdd = ((cum_wealth - cum_wealth.cummax()) / cum_wealth.cummax()).min()
    return {"Portfolio": name, "Cum Ret": f"{cum_return*100:.2f}%", "Sharpe": f"{sharpe:.2f}", "MDD": f"{mdd*100:.2f}%"}

def main():
    df_returns = pd.read_csv("chip4_usd_returns.csv", index_col='Date', parse_dates=True)
    all_results = []

    for date_str, solvers in portfolios_data.items():
        print(f"\n📅 分析日期: {date_str}")
        target_date = pd.to_datetime(date_str)
        oos_data = df_returns.loc[target_date:].iloc[1:OOS_DAYS+1]
        
        plt.figure(figsize=(10, 6))
        for solver_name, tickers in solvers.items():
            valid = [t for t in tickers if t in oos_data.columns]
            if not valid: continue
            
            port_ret = oos_data[valid].mean(axis=1)
            metrics = calculate_metrics(port_ret, solver_name)
            metrics["Date"] = date_str
            # 新增記錄運算耗時欄位供備查
            metrics["Execution Time (s)"] = execution_times.get(date_str, {}).get(solver_name, "N/A")
            all_results.append(metrics)
            
            cum_wealth = (1 + port_ret).cumprod() * 100
            plt.plot(cum_wealth.index, cum_wealth, label=f"{solver_name}", **SOLVERS[solver_name])

        plt.title(f"Performance Comparison: {date_str}")
        plt.legend()
        plt.savefig(f"backtest_{date_str}.png")
        plt.close()

    pd.DataFrame(all_results).to_csv("summary_backtest.csv", index=False)
    print("\n✅ 回測完成，結果已存入 summary_backtest.csv")

if __name__ == "__main__":
    main()