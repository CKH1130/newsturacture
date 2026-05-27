import csv
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


TARGET_DATES = [
    "2019-05-13",
    "2019-06-19",
    "2019-06-24",
    "2020-03-04",
    "2026-04-08",
]

BACKTEST_FILE = Path("backtest_metrics_all_dates.csv")
QUALITY_FILE = Path("evaluation_quality_table.csv")
OUTPUT_FILE = Path("summary_performance_table.md")

SA_KEY = "SA"
QAOA_KEY = "QAOA"

METRICS = [
    {
        "name": "平均累積報酬率",
        "column": "Cumulative Return",
        "unit": "%",
        "qaoa_wins": lambda qaoa, sa: qaoa > sa,
    },
    {
        "name": "平均年化波動率",
        "column": "Ann. Volatility",
        "unit": "%",
        "qaoa_wins": lambda qaoa, sa: qaoa < sa,
    },
    {
        "name": "平均最大回撤 (MDD)",
        "column": "Max Drawdown",
        "unit": "%",
        "qaoa_wins": lambda qaoa, sa: qaoa > sa,
    },
    {
        "name": "平均夏普值 (Sharpe Ratio)",
        "column": "Sharpe Ratio",
        "unit": "",
        "qaoa_wins": lambda qaoa, sa: qaoa > sa,
    },
]


def parse_number(value):
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    return float(text)


def portfolio_key(portfolio_name):
    if "QAOA" in portfolio_name:
        return QAOA_KEY
    if "SA" in portfolio_name or "古典" in portfolio_name:
        return SA_KEY
    raise ValueError(f"無法辨識 Portfolio 類別: {portfolio_name}")


def load_backtest_metrics(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，請先執行 10backtest_engine.py 產生回測總表。")

    data = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["Date"]
            if date not in TARGET_DATES:
                continue

            key = portfolio_key(row["Portfolio"])
            data[(date, key)] = {
                metric["column"]: parse_number(row[metric["column"]])
                for metric in METRICS
            }

    missing = [
        f"{date} {key}"
        for date in TARGET_DATES
        for key in (SA_KEY, QAOA_KEY)
        if (date, key) not in data
    ]
    if missing:
        raise ValueError(f"回測資料缺少以下列：{', '.join(missing)}")

    return data


def load_violation_counts(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，請先執行 8evaluation_engine.py 產生違規統計。")

    violations = {SA_KEY: {}, QAOA_KEY: {}}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["Date"]
            if date not in TARGET_DATES:
                continue

            violations[SA_KEY][date] = int(parse_number(row["SA_Viol"]))
            violations[QAOA_KEY][date] = int(parse_number(row["QAOA_Viol"]))

    missing = [
        f"{date} {key}"
        for date in TARGET_DATES
        for key in (SA_KEY, QAOA_KEY)
        if date not in violations[key]
    ]
    if missing:
        raise ValueError(f"違規資料缺少以下列：{', '.join(missing)}")

    return violations


def average_metric(data, key, column):
    values = [data[(date, key)][column] for date in TARGET_DATES]
    return sum(values) / len(values)


def qaoa_win_rate(data, metric):
    wins = 0
    for date in TARGET_DATES:
        qaoa_value = data[(date, QAOA_KEY)][metric["column"]]
        sa_value = data[(date, SA_KEY)][metric["column"]]
        if metric["qaoa_wins"](qaoa_value, sa_value):
            wins += 1

    return wins, wins / len(TARGET_DATES) * 100


def zero_violation_ratio(violations, key):
    zero_count = sum(1 for date in TARGET_DATES if violations[key][date] == 0)
    ratio = zero_count / len(TARGET_DATES) * 100
    return zero_count, ratio


def format_metric(value, unit):
    if unit == "%":
        return f"{value:.2f}%"
    return f"{value:.2f}"


def format_win_rate(wins, total=5):
    return f"{wins} / {total} ({wins / total * 100:.0f}%)"


def format_zero_ratio(zero_count, total=5):
    return f"{zero_count / total * 100:.0f}% ({zero_count}/{total})"


def build_summary_table():
    backtest_data = load_backtest_metrics(BACKTEST_FILE)
    violations = load_violation_counts(QUALITY_FILE)

    rows = [
        [
            "綜合績效與約束指標 (5 期平均)",
            "傳統古典最佳解 (模擬退火 SA)",
            "量子演算法次佳解 (QAOA)",
            "QAOA 對決 SA 勝率 (Win Rate)",
        ],
        [
            ":---",
            ":---",
            ":---",
            ":---",
        ],
    ]

    for metric in METRICS:
        sa_avg = average_metric(backtest_data, SA_KEY, metric["column"])
        qaoa_avg = average_metric(backtest_data, QAOA_KEY, metric["column"])
        qaoa_wins, _ = qaoa_win_rate(backtest_data, metric)

        rows.append([
            f"**{metric['name']}**",
            format_metric(sa_avg, metric["unit"]),
            format_metric(qaoa_avg, metric["unit"]),
            format_win_rate(qaoa_wins),
        ])

    sa_zero_count, _ = zero_violation_ratio(violations, SA_KEY)
    qaoa_zero_count, _ = zero_violation_ratio(violations, QAOA_KEY)
    rows.append([
        "**零違規比例 (完全滿足實務限制)**",
        format_zero_ratio(sa_zero_count),
        format_zero_ratio(qaoa_zero_count),
        "0 / 5 (0%)",
    ])

    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in rows
    )


def main():
    table = build_summary_table()
    print(table)
    OUTPUT_FILE.write_text(table + "\n", encoding="utf-8")
    print(f"\n表格已儲存為: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
