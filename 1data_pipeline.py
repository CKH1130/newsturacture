# data_pipeline.py

import os
from datetime import datetime

import pandas as pd
import yfinance as yf


def download_adj_close(tickers, start_date, end_date):
    print(f"[Download] 下載 {start_date} ~ {end_date} 的資料中...")

    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False
    )

    if df.empty:
        raise RuntimeError("下載結果為空，請檢查 ticker 或網路連線。")

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" not in df.columns.get_level_values(0):
            raise RuntimeError("下載結果中找不到 'Adj Close'。")
        adj_close = df["Adj Close"].copy()
    else:
        if "Adj Close" not in df.columns:
            raise RuntimeError("下載結果中找不到 'Adj Close'。")
        adj_close = df[["Adj Close"]].copy()
        if len(tickers) == 1:
            adj_close.columns = tickers

    adj_close.index = pd.to_datetime(adj_close.index)
    adj_close.index.name = "Date"
    adj_close = adj_close.sort_index()

    return adj_close


def validate_data(df, name):
    if df.empty:
        raise RuntimeError(f"{name} 是空的。")

    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        raise RuntimeError(f"{name} 以下欄位全為缺值：{all_nan_cols}")

    print(f"[Check] {name} 形狀: {df.shape}")
    print(f"[Check] {name} 日期區間: {df.index.min().date()} ~ {df.index.max().date()}")


def convert_to_usd(df_assets_local, df_fx):
    df_usd = df_assets_local.copy()

    # Yahoo Finance 匯率：
    # TWD=X, JPY=X, KRW=X 代表 1 USD = 幾單位當地貨幣
    # 所以當地貨幣股價要除以匯率，才能換成 USD

    tw_tickers = ["2330.TW", "2454.TW", "3711.TW"]
    two_tickers = ["6488.TWO"]   # 櫃買
    jp_tickers = ["8035.T", "6857.T", "4063.T"]
    kr_tickers = ["005930.KS", "000660.KS", "042700.KS"]

    for t in tw_tickers + two_tickers:
        df_usd[t] = df_usd[t] / df_fx["TWD=X"]

    for t in jp_tickers:
        df_usd[t] = df_usd[t] / df_fx["JPY=X"]

    for t in kr_tickers:
        df_usd[t] = df_usd[t] / df_fx["KRW=X"]

    return df_usd


def main():
    print("=== 啟動 Chip 4 跨國半導體資料收集管線 ===")

    tickers_us = ["NVDA", "AMD", "QCOM", "AMAT", "ASML"]

    # 台股上市 .TW；櫃買 .TWO
    tickers_tw = ["2330.TW", "2454.TW", "3711.TW"]
    tickers_two = ["6488.TWO"]

    tickers_jp = ["8035.T", "6857.T", "4063.T"]
    tickers_kr = ["005930.KS", "000660.KS", "042700.KS"]

    fx_tickers = ["TWD=X", "JPY=X", "KRW=X"]

    asset_tickers = tickers_us + tickers_tw + tickers_two + tickers_jp + tickers_kr
    all_tickers = asset_tickers + fx_tickers

    start_date = "2019-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"[Step 1] 下載期間：{start_date} ~ {end_date}")

    df_all = download_adj_close(all_tickers, start_date, end_date)

    # 先做前向填補，處理跨市場休市不同步
    df_all = df_all.ffill()

    df_assets_local = df_all[asset_tickers].copy()
    df_fx = df_all[fx_tickers].copy()

    validate_data(df_assets_local, "原幣別資產價格")
    validate_data(df_fx, "匯率資料")

    # 轉成美元
    df_assets_usd = convert_to_usd(df_assets_local, df_fx)

    # 再補一次，最後去掉仍有缺值的列
    df_assets_usd = df_assets_usd.ffill().dropna(how="any")
    validate_data(df_assets_usd, "美元價格")

    # 簡單報酬率
    df_returns = df_assets_usd.pct_change().dropna(how="any")
    validate_data(df_returns, "美元報酬率")

    df_assets_usd.to_csv("chip4_usd_prices.csv", encoding="utf-8-sig")
    df_returns.to_csv("chip4_usd_returns.csv", encoding="utf-8-sig")

    print("\n=== 完成 ===")
    print("已輸出：chip4_usd_prices.csv")
    print("已輸出：chip4_usd_returns.csv")


if __name__ == "__main__":
    main()