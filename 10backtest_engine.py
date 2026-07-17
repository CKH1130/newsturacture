import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# === 參數設定 ===
# 對應新版論文 3.8 節，調倉日後 20 個交易日
OOS_DAYS = 20 
RISK_FREE_RATE = 0.02  # 假設年化無風險利率 2%

# 全體 15 檔候選標的
ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

# 移除 SA，對齊論文表 3-2 的對照組名稱
PLOT_STYLES = {
    "Brute Force (全域絕對最佳)": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 2.5},
    "QAOA (IBM 真實量子電腦)": {"color": "#d62728", "marker": "s", "linestyle": "--", "linewidth": 2.5},
}

# === 投資組合輸入區 ===
# 🚨 修正 3：這裡的範例已改為符合 K=4 與 (1,1,1,1) 配置。
# ⚠️ 請務必替換為您執行 Script 5 (Brute Force) 與 Script 9 (IBM QAOA) 產出的真實名單！
這邊重跑之後要更改
portfolios_20190513 =   {
    "Brute Force (全域絕對最佳)": ['QCOM', '6488.TWO', '8035.T', '000660.KS'],
    "QAOA (IBM 真實量子電腦)": ['AMD', '6488.TWO', '4063.T', '042700.KS']
}

portfolios_20190619 = {  
    "Brute Force (全域絕對最佳)": ['QCOM', '6488.TWO', '4063.T', '005930.KS'],
    "QAOA (IBM 真實量子電腦)": ['QCOM', '3711.TW', '6857.T', '042700.KS']
}

portfolios_20190624 = {  
    "Brute Force (全域絕對最佳)": ['QCOM', '6488.TWO', '4063.T', '042700.KS'],
    "QAOA (IBM 真實量子電腦)": ['NVDA', '6488.TWO', '8035.T', '000660.KS']
}

portfolios_20200304 = {  
    "Brute Force (全域絕對最佳)": ['ASML', '2454.TW', '6857.T', '042700.KS'],
    "QAOA (IBM 真實量子電腦)": ['ASML', '3711.TW', '4063.T', '042700.KS']
}

portfolios_20260427 = {
    "Brute Force (全域絕對最佳)": ['QCOM', '6488.TWO', '8035.T', '005930.KS'],
    "QAOA (IBM 真實量子電腦)": ['AMD', '6488.TWO', '6857.T', '000660.KS'] 
}

target_dates = {
    "2019-05-13": portfolios_20190513,  # 極端大跌日
    "2019-06-19": portfolios_20190619,  # 極端大漲日
    "2019-06-24": portfolios_20190624,  # 平穩日
    "2020-03-04": portfolios_20200304,  # 高波動日
    "2026-04-27": portfolios_20260427   # 事件衝擊日
}

def calculate_metrics(daily_returns, portfolio_name):
    """計算學術級財務績效指標"""
    # 累積報酬率
    cum_return = (1 + daily_returns).prod() - 1
    ann_return = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol != 0 else 0
    
    cum_wealth = (1 + daily_returns).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    mdd = drawdown.min()
    
    return {
        "Portfolio": portfolio_name,
        "Cumulative Return": f"{cum_return*100:.2f}%",
        "Ann. Return": f"{ann_return*100:.2f}%",
        "Ann. Volatility": f"{ann_vol*100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{mdd*100:.2f}%"
    }

def format_markdown_table(headers, rows):
    """簡易 Markdown 表格輸出"""
    if not rows:
        return "無資料"

    rows = [[str(value) for value in row] for row in rows]
    headers = [str(header) for header in headers]
    col_widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def markdown_row(values):
        return "| " + " | ".join(
            str(value).ljust(col_widths[i]) for i, value in enumerate(values)
        ) + " |"

    lines = [
        markdown_row(headers),
        "| " + " | ".join("-" * width for width in col_widths) + " |",
    ]
    lines.extend(markdown_row(row) for row in rows)
    return "\n".join(lines)

def main():
    print("=== 🚀 啟動樣本外財務績效回測引擎 (Out-of-sample Backtester) ===\n")
    
    try:
        df_returns = pd.read_csv("chip4_usd_returns.csv", index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("找不到 'chip4_usd_returns.csv'，請確認檔案是否存在。")
        return

    all_metrics = []

    for target_date_str, portfolios in target_dates.items():
        print(f"📅 正在執行調倉日: {target_date_str} 之回測 (期間: 往後 {OOS_DAYS} 個交易日)")
        
        target_date = pd.to_datetime(target_date_str)
        
        # 篩選樣本外期間資料
        oos_data = df_returns.loc[target_date:].iloc[1:OOS_DAYS+1] 
        
        if len(oos_data) == 0:
            print(f"⚠️ 警告: 找不到 {target_date_str} 之後的資料。")
            continue
            
        metrics_list = []
        plt.figure(figsize=(10, 6))
        
        for name, tickers in portfolios.items():
            valid_tickers = [t for t in tickers if t in oos_data.columns]
            if not valid_tickers:
                print(f"⚠️ 警告: {name} 沒有可用 ticker，已跳過。")
                continue
            
            # 等權重每日報酬率
            port_daily_return = oos_data[valid_tickers].mean(axis=1)
            
            metrics = calculate_metrics(port_daily_return, name)
            metrics["Date"] = target_date_str
            metrics_list.append(metrics)
            all_metrics.append(metrics)
            
            # 畫圖
            cum_wealth = (1 + port_daily_return).cumprod() * 100 
            style = PLOT_STYLES.get(name, {"linewidth": 2.5})
            plt.plot(
                cum_wealth.index,
                cum_wealth,
                label=f"{name} (Ret: {metrics['Cumulative Return']}, Sharpe: {metrics['Sharpe Ratio']})",
                markevery=max(len(cum_wealth) // 5, 1),
                markersize=5,
                **style
            )

        if not metrics_list:
            plt.close()
            continue
        
        plt.title(f"Out-of-sample Cumulative Wealth (Rebalance Date: {target_date_str})", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Wealth (Base = 100)", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_name = f"backtest_plot_{target_date_str}.png"
        plt.savefig(img_name, dpi=300)
        plt.close()
        print(f"📉 走勢圖已儲存為: {img_name}")
        
        df_metrics = pd.DataFrame(metrics_list)
        print("\n" + "="*80)
        print(f"📋 各求解器投資組合之樣本外財務績效比較 (基準日: {target_date_str})")
        print("="*80)
        print(format_markdown_table(df_metrics.columns.tolist(), df_metrics.values.tolist()))
        print("="*80 + "\n")

    if all_metrics:
        df_all_metrics = pd.DataFrame(all_metrics)
        cols = ["Date"] + [col for col in df_all_metrics.columns if col != "Date"]
        df_all_metrics = df_all_metrics[cols]
        df_all_metrics.to_csv("backtest_metrics_all_dates.csv", index=False, encoding="utf-8-sig")

        print("\n" + "="*80)
        print("📋 五個事件日之樣本外財務績效總表")
        print("="*80)
        print(format_markdown_table(df_all_metrics.columns.tolist(), df_all_metrics.values.tolist()))
        print("總表已儲存為: backtest_metrics_all_dates.csv")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()