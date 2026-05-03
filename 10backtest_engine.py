import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === 參數設定 ===
# 設定回測期間 (調倉日後 21 個交易日，約 1 個月)
OOS_DAYS = 21 
RISK_FREE_RATE = 0.02  # 假設年化無風險利率 2%

# 全體 15 檔候選標的
ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

PLOT_STYLES = {
    "SA/BF (古典絕對最佳)": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 2.5},
    "QAOA (IBM真機/硬體雜訊)": {"color": "#d62728", "marker": "s", "linestyle": "--", "linewidth": 2.5},
}

# === 投資組合輸入區 (請替換為您各階段選出的真實名單) ===
# 版本 A：不放 Benchmark，聚焦求解器之間的樣本外績效比較
portfolios_20260408 = {
    "SA/BF (古典絕對最佳)": ['ASML', '2330.TW', '6488.TWO', '4063.T', '042700.KS'],
    "QAOA (IBM真機/硬體雜訊)": ['NVDA', '2330.TW', '8035.T', '000660.KS', 'ASML'] # 替換成您真機的結果
}

# 若要跑其他天 (例如 2019-05-13)，可依樣畫葫蘆建立字典
target_dates = {
    "2026-04-08": portfolios_20260408
}

def calculate_metrics(daily_returns, portfolio_name):
    """計算學術級財務績效指標"""
    # 累積報酬率
    cum_return = (1 + daily_returns).prod() - 1
    
    # 年化報酬率 (假設一年 252 個交易日)
    ann_return = daily_returns.mean() * 252
    
    # 年化波動率
    ann_vol = daily_returns.std() * np.sqrt(252)
    
    # 夏普值 (Sharpe Ratio)
    sharpe_ratio = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol != 0 else 0
    
    # 最大回撤 (Maximum Drawdown, MDD)
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
    """不依賴 tabulate 的簡易 Markdown 表格輸出。"""
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
    
    # 1. 讀取歷史日報酬率資料
    try:
        df_returns = pd.read_csv("chip4_usd_returns.csv", index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print("找不到 'chip4_usd_returns.csv'，請確認檔案是否存在。")
        return

    # 2. 針對每個調倉日進行回測
    for target_date_str, portfolios in target_dates.items():
        print(f"📅 正在執行調倉日: {target_date_str} 之回測 (期間: 往後 {OOS_DAYS} 個交易日)")
        
        target_date = pd.to_datetime(target_date_str)
        
        # 篩選樣本外 (Out-of-sample) 期間資料
        oos_data = df_returns.loc[target_date:].iloc[1:OOS_DAYS+1] 
        
        if len(oos_data) == 0:
            print(f"⚠️ 警告: 找不到 {target_date_str} 之後的資料。")
            continue
            
        metrics_list = []
        plt.figure(figsize=(10, 6))
        
        # 3. 計算各投資組合績效與畫圖
        for name, tickers in portfolios.items():
            # 確保選出的股票都在資料內
            valid_tickers = [t for t in tickers if t in oos_data.columns]
            if not valid_tickers:
                print(f"⚠️ 警告: {name} 沒有可用 ticker，已跳過。")
                continue
            
            # 計算等權重每日報酬率 (Daily returns of equal-weighted portfolio)
            port_daily_return = oos_data[valid_tickers].mean(axis=1)
            
            # 算指標
            metrics = calculate_metrics(port_daily_return, name)
            metrics_list.append(metrics)
            
            # 畫累積報酬走勢圖
            cum_wealth = (1 + port_daily_return).cumprod() * 100 # 基期 100
            style = PLOT_STYLES.get(name, {"linewidth": 2.5})
            plt.plot(
                cum_wealth.index,
                cum_wealth,
                label=f"{name} (Ret: {metrics['Cumulative Return']}, Sharpe: {metrics['Sharpe Ratio']})",
                markevery=max(len(cum_wealth) // 5, 1),
                markersize=5,
                **style
            )
        
        # === 畫圖設定 (符合論文格式) ===
        plt.title(f"Solver Portfolio Out-of-sample Cumulative Wealth (Rebalance Date: {target_date_str})", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Wealth (Base = 100)", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 存檔圖片
        img_name = f"backtest_plot_{target_date_str}.png"
        plt.savefig(img_name, dpi=300)
        plt.close()
        print(f"📉 走勢圖已儲存為: {img_name}")
        
        # === 印出表格 (可直接貼入論文表三) ===
        df_metrics = pd.DataFrame(metrics_list)
        print("\n" + "="*80)
        print(f"📋 表三：各求解器投資組合之樣本外財務績效比較 (基準日: {target_date_str})")
        print("="*80)
        print(format_markdown_table(df_metrics.columns.tolist(), df_metrics.values.tolist()))
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
