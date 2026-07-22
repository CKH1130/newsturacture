import math
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


BACKTEST_ENGINE_FILE = Path("10backtest_engine.py")
RETURNS_FILE = Path("chip4_usd_returns.csv")
SUMMARY_CSV_FILE = Path("summary_backtest.csv")
WIDE_CSV_FILE = Path("summary_daily_comparison.csv")
MARKDOWN_FILE = Path("summary_performance_table.md")

METHODS = [
    "Brute Force",
    "QAOA (Ideal)",
    "QAOA (Noisy)",
    "QAOA (IBM Real)",
]

METHOD_LABELS = {
    "Brute Force": "窮舉法",
    "QAOA (Ideal)": "理想模擬",
    "QAOA (Noisy)": "雜訊模擬",
    "QAOA (IBM Real)": "真實量子電腦",
}

MARKETS = {
    "US": ["NVDA", "AMD", "QCOM", "AMAT", "ASML"],
    "TW": ["2330.TW", "2454.TW", "3711.TW", "6488.TWO"],
    "JP": ["8035.T", "6857.T", "4063.T"],
    "KR": ["005930.KS", "000660.KS", "042700.KS"],
}

DEPENDENCIES = [
    ("NVDA", "2330.TW"),          # X_NVIDIA(1 - X_TSMC)
    ("AMD", "2330.TW"),          # X_AMD(1 - X_TSMC)
    ("QCOM", "2330.TW"),          # X_Qualcomm(1 - X_TSMC)
    ("2330.TW", "ASML"),          # X_TSMC(1 - X_ASML)
    ("2330.TW", "6488.TWO"),      # X_TSMC(1 - X_GlobalWafers) [環球晶]
    ("005930.KS", "2330.TW")      # X_Samsung(1 - X_TSMC) [三星]
]


def load_backtest_config():
    if not BACKTEST_ENGINE_FILE.exists():
        raise FileNotFoundError(f"找不到 {BACKTEST_ENGINE_FILE}")

    namespace = runpy.run_path(str(BACKTEST_ENGINE_FILE))
    portfolios_data = namespace.get("portfolios_data")
    if not portfolios_data:
        raise ValueError(f"{BACKTEST_ENGINE_FILE} 裡找不到 portfolios_data")

    missing = [
        f"{date} / {method}"
        for date, portfolios in portfolios_data.items()
        for method in METHODS
        if method not in portfolios
    ]
    if missing:
        raise ValueError("投組資料缺少以下方法：" + ", ".join(missing))

    # ✅ 新增：讀取第 10 段的運算耗時資料 (若無則預設為空字典)
    execution_times = namespace.get("execution_times", {})

    return {
        "portfolios_data": portfolios_data,
        "execution_times": execution_times,
        "oos_days": int(namespace.get("OOS_DAYS", 20)),
        "risk_free_rate": float(namespace.get("RISK_FREE_RATE", 0.02)),
    }


def load_returns():
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(f"找不到 {RETURNS_FILE}")
    return pd.read_csv(RETURNS_FILE, index_col="Date", parse_dates=True)


def validate_portfolio(selected_tickers):
    violations = abs(len(selected_tickers) - 4)

    for tickers in MARKETS.values():
        market_count = sum(ticker in tickers for ticker in selected_tickers)
        violations += abs(market_count - 1)

    violations += sum(
        dep in selected_tickers and relies_on not in selected_tickers
        for dep, relies_on in DEPENDENCIES
    )
    return int(violations)


def calculate_metrics(daily_returns, risk_free_rate):
    if daily_returns.empty:
        raise ValueError("回測區間沒有任何交易日資料")

    cumulative_return = (1 + daily_returns).prod() - 1
    annual_return = daily_returns.mean() * 252
    annual_volatility = daily_returns.std() * math.sqrt(252)
    sharpe_ratio = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility and not np.isnan(annual_volatility)
        else 0.0
    )

    cumulative_wealth = (1 + daily_returns).cumprod()
    drawdown = (cumulative_wealth - cumulative_wealth.cummax()) / cumulative_wealth.cummax()
    max_drawdown = drawdown.min()

    return {
        "trading_days": int(daily_returns.count()),
        "cumulative_return": float(cumulative_return),
        "annual_volatility": float(annual_volatility),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
    }


def format_percent(value):
    return f"{value * 100:.2f}%"


def format_number(value):
    return f"{value:.2f}"


def format_zero_violation(violations):
    return f"{'是' if violations == 0 else '否'} ({violations})"


def build_daily_results(config, df_returns):
    results = []

    for date_str, portfolios in config["portfolios_data"].items():
        target_date = pd.to_datetime(date_str)
        oos_data = df_returns.loc[target_date:].iloc[1 : config["oos_days"] + 1]

        if oos_data.empty:
            raise ValueError(f"{date_str} 之後沒有 out-of-sample 回測資料")

        for method in METHODS:
            selected_tickers = list(portfolios[method])
            valid_tickers = [ticker for ticker in selected_tickers if ticker in oos_data.columns]
            if not valid_tickers:
                raise ValueError(f"{date_str} / {method} 沒有可用 ticker：{selected_tickers}")

            daily_returns = oos_data[valid_tickers].mean(axis=1)
            metrics = calculate_metrics(daily_returns, config["risk_free_rate"])
            violations = validate_portfolio(selected_tickers)

            # ✅ 抓取對應日期的運算耗時
            exec_time = config["execution_times"].get(date_str, {}).get(method, "N/A")

            results.append(
                {
                    "Date": date_str,
                    "Portfolio": method,
                    "Method": METHOD_LABELS[method],
                    "Selected Tickers": ", ".join(selected_tickers),
                    "Valid Tickers": ", ".join(valid_tickers),
                    "Trading Days": metrics["trading_days"],
                    "Cumulative Return": format_percent(metrics["cumulative_return"]),
                    "Ann. Volatility": format_percent(metrics["annual_volatility"]),
                    "Max Drawdown": format_percent(metrics["max_drawdown"]),
                    "Sharpe Ratio": format_number(metrics["sharpe_ratio"]),
                    "Violations": violations,
                    "Zero Violation": "Yes" if violations == 0 else "No",
                    "Execution Time (s)": exec_time,  # ✅ 紀錄耗時
                    "_cumulative_return": metrics["cumulative_return"],
                    "_annual_volatility": metrics["annual_volatility"],
                    "_max_drawdown": metrics["max_drawdown"],
                    "_sharpe_ratio": metrics["sharpe_ratio"],
                }
            )

    return results


def build_wide_rows(results):
    by_date_method = {
        (row["Date"], row["Portfolio"]): row
        for row in results
    }
    dates = sorted({row["Date"] for row in results})

    # ✅ 指標清單加入「運算耗時 (秒)」
    metric_specs = [
        ("累積報酬率", lambda row: row["Cumulative Return"]),
        ("年化波動率", lambda row: row["Ann. Volatility"]),
        ("最大回撤", lambda row: row["Max Drawdown"]),
        ("夏普值", lambda row: row["Sharpe Ratio"]),
        ("零違規", lambda row: format_zero_violation(row["Violations"])),
        ("運算耗時 (秒)", lambda row: str(row["Execution Time (s)"])),  # ✅ 呈現運算耗時
    ]

    wide_rows = []
    for date in dates:
        for metric_name, getter in metric_specs:
            row = {"Date": date, "Metric": metric_name}
            for method in METHODS:
                row[METHOD_LABELS[method]] = getter(by_date_method[(date, method)])
            wide_rows.append(row)
    return wide_rows


def build_markdown_table(wide_rows):
    headers = ["日期", "指標"] + [METHOD_LABELS[method] for method in METHODS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |",
    ]

    for row in wide_rows:
        values = [row["Date"], row["Metric"]] + [
            row[METHOD_LABELS[method]]
            for method in METHODS
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_outputs(results, wide_rows):
    public_columns = [
        "Date",
        "Portfolio",
        "Method",
        "Selected Tickers",
        "Valid Tickers",
        "Trading Days",
        "Cumulative Return",
        "Ann. Volatility",
        "Max Drawdown",
        "Sharpe Ratio",
        "Violations",
        "Zero Violation",
        "Execution Time (s)",  # ✅ 加入導出的 CSV 欄位
    ]
    pd.DataFrame(results)[public_columns].to_csv(
        SUMMARY_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(wide_rows).to_csv(
        WIDE_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    markdown_table = build_markdown_table(wide_rows)
    MARKDOWN_FILE.write_text(markdown_table + "\n", encoding="utf-8")
    return markdown_table


def main():
    config = load_backtest_config()
    df_returns = load_returns()
    results = build_daily_results(config, df_returns)
    wide_rows = build_wide_rows(results)
    markdown_table = write_outputs(results, wide_rows)

    print(markdown_table)
    print(f"\n已輸出：{SUMMARY_CSV_FILE}")
    print(f"已輸出：{WIDE_CSV_FILE}")
    print(f"已輸出：{MARKDOWN_FILE}")


if __name__ == "__main__":
    main()