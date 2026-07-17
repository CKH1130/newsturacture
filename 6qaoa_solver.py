import csv
import json
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
import warnings

warnings.filterwarnings("ignore")

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    USE_AER = True
except ImportError:
    from qiskit.primitives import StatevectorSampler as AerSampler
    USE_AER = False


ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

TARGET_DATES = [
    "2019-05-13",
    "2019-06-19",
    "2019-06-24",
    "2020-03-04",
    "2026-04-27",
]

# 🚨 警告：這裡必須填入「第五段：古典暴力破解(0602版)」跑出的最新 Energy！
# 如果沒有填入正確數值，Optimality Gap 將無法計算。
BRUTE_FORCE_ENERGY_BY_DATE = {
    "2019-05-13": -0.008988, # 替換為新算出的 Brute Force Energy
    "2019-06-19": -0.004355, # 替換為新算出的 Brute Force Energy
    "2019-06-24": -0.023807, # 替換為新算出的 Brute Force Energy
    "2020-03-04": -0.006620, # 替換為新算出的 Brute Force Energy
    "2026-04-27": 0.004929, # 替換為新算出的 Brute Force Energy
}


def load_json_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def build_inputs_for_date(date, mu_data, sigma_data, assets):
    if date not in mu_data:
        raise ValueError(f"mu_data 找不到日期 {date}")
    if date not in sigma_data:
        raise ValueError(f"sigma_data 找不到日期 {date}")

    missing_mu = [ticker for ticker in assets if ticker not in mu_data[date]]
    if missing_mu:
        raise ValueError(f"mu 缺少 ticker: {missing_mu}")

    mu = np.array([mu_data[date][ticker] for ticker in assets], dtype=float)
    sigma = np.array(sigma_data[date], dtype=float)

    if sigma.shape != (len(assets), len(assets)):
        raise ValueError(f"Sigma 矩陣形狀錯誤: {sigma.shape}")

    if not np.isfinite(mu).all():
        bad_tickers = [assets[i] for i, value in enumerate(mu) if not np.isfinite(value)]
        raise ValueError(f"mu 含非有限值: {bad_tickers}")

    if not np.isfinite(sigma).all():
        bad_count = int((~np.isfinite(sigma)).sum())
        raise ValueError(f"Sigma 含 {bad_count} 個非有限值")

    return mu, sigma


def build_chip4_qubo(mu, sigma, lmbda=0.5, p1=50.0, p3=10.0):
    qp = QuadraticProgram(name="Chip4_QAOA")
    assets = ASSETS

    for asset in assets:
        qp.binary_var(name=asset)

    linear = {assets[i]: -(1 - lmbda) * mu[i] for i in range(len(assets))}
    quadratic = {}

    for i in range(len(assets)):
        for j in range(len(assets)):
            quadratic[(assets[i], assets[j])] = lmbda * sigma[i][j]

    dependencies = [
        ("NVDA", "2330.TW"),
        ("AMD", "2330.TW"),
        ("2330.TW", "ASML")
    ]

    for dep, relies_on in dependencies:
        linear[dep] = linear.get(dep, 0.0) + p3
        key = (dep, relies_on)
        quadratic[key] = quadratic.get(key, 0.0) - p3

    qp.minimize(linear=linear, quadratic=quadratic)

    markets = {
        "US": (["NVDA", "AMD", "QCOM", "AMAT", "ASML"], 1),
        "TW": (["2330.TW", "2454.TW", "3711.TW", "6488.TWO"], 1), 
        "JP": (["8035.T", "6857.T", "4063.T"], 1),
        "KR": (["005930.KS", "000660.KS", "042700.KS"], 1)
    }

    for market_name, (tickers, required_count) in markets.items():
        qp.linear_constraint(
            linear={ticker: 1 for ticker in tickers},
            sense="==",
            rhs=required_count,
            name=f"Mkt_{market_name}"
        )

    converter = QuadraticProgramToQubo(penalty=p1)
    qubo = converter.convert(qp)

    return qubo, assets


def calculate_original_energy(selected_tickers, mu, sigma, assets, lmbda=0.5, p3=10.0):
    x = np.zeros(len(assets))

    for ticker in selected_tickers:
        idx = assets.index(ticker)
        x[idx] = 1

    risk = float(x.T @ sigma @ x)
    expected_return = float(mu.T @ x)
    h_rr = lmbda * risk - (1 - lmbda) * expected_return

    dependencies = [
        ("NVDA", "2330.TW"),
        ("AMD", "2330.TW"),
        ("2330.TW", "ASML")
    ]

    violations = 0
    for dep, relies_on in dependencies:
        if dep in selected_tickers and relies_on not in selected_tickers:
            violations += 1

    h_dep = p3 * violations
    total_energy = h_rr + h_dep

    return total_energy, risk, expected_return, violations


def build_qaoa_optimizer(date):
    def qaoa_callback(eval_count, parameters, mean, metadata):
        print(f"[{date} QAOA] eval={eval_count:03d}, energy={mean:.6f}", flush=True)

    if USE_AER:
        backend = AerSimulator(method="statevector")
        sampler = AerSampler()
        transpiler = generate_preset_pass_manager(
            optimization_level=1,
            backend=backend
        )

        print(
            "使用 qiskit_aer.primitives.SamplerV2 + AerSimulator transpiler",
            flush=True
        )

        qaoa = QAOA(
            sampler=sampler,
            optimizer=COBYLA(maxiter=30),
            reps=1,
            callback=qaoa_callback,
            transpiler=transpiler
        )
    else:
        sampler = AerSampler()

        print("使用 qiskit.primitives.StatevectorSampler", flush=True)

        qaoa = QAOA(
            sampler=sampler,
            optimizer=COBYLA(maxiter=5),
            reps=1,
            callback=qaoa_callback
        )

    return MinimumEigenOptimizer(qaoa)


def solve_for_date(target_date, mu_data, sigma_data):
    print(f"載入調倉日: {target_date} 之參數...", flush=True)

    mu, sigma = build_inputs_for_date(target_date, mu_data, sigma_data, ASSETS)

    qubo_model, assets = build_chip4_qubo(
        mu=mu,
        sigma=sigma,
        lmbda=0.5,
        p1=50.0,
        p3=10.0
    )

    print("QUBO 模型轉換完畢", flush=True)
    print(f"總變數數量: {qubo_model.get_num_vars()}", flush=True)
    print(f"總限制式數量: {qubo_model.get_num_linear_constraints()}", flush=True)
    print("", flush=True)

    print("初始化 QAOA：reps=1, Aer COBYLA maxiter=30 / fallback maxiter=5", flush=True)
    optimizer = build_qaoa_optimizer(target_date)

    print("\n即將進入 optimizer.solve(qubo_model)...", flush=True)
    result = optimizer.solve(qubo_model)
    print("optimizer.solve() 已完成。\n", flush=True)

    selected_indices = [
        i for i, value in enumerate(result.x[:len(ASSETS)])
        if round(value) == 1
    ]
    selected_tickers = [assets[i] for i in selected_indices]

    original_energy, risk, expected_return, violations = calculate_original_energy(
        selected_tickers=selected_tickers,
        mu=mu,
        sigma=sigma,
        assets=assets,
        lmbda=0.5,
        p3=10.0
    )

    brute_force_energy = BRUTE_FORCE_ENERGY_BY_DATE.get(target_date)
    gap = None
    if brute_force_energy is not None and brute_force_energy != 0:
        gap = abs(original_energy - brute_force_energy) / abs(brute_force_energy)

    return {
        "date": target_date,
        "selected_tickers": selected_tickers,
        "selected_count": len(selected_tickers),
        "qaoa_qubo_fval": float(result.fval),
        "original_energy": original_energy,
        "brute_force_energy": brute_force_energy,
        "optimality_gap": gap,
        "expected_return": expected_return,
        "risk": risk,
        "dependency_violations": violations
    }


def print_result(result):
    print("==================================================", flush=True)
    print("QAOA 求解完成", flush=True)
    print(f"調倉日: {result['date']}", flush=True)
    print(f"QAOA 找到之投資組合: {result['selected_tickers']}", flush=True)
    print(f"選股數量: {result['selected_count']}", flush=True)
    print("", flush=True)
    print(f"QAOA QUBO fval: {result['qaoa_qubo_fval']:.6f}", flush=True)
    print(f"重新代回原始目標函數 Energy: {result['original_energy']:.6f}", flush=True)
    if result["brute_force_energy"] is None:
        print("Brute Force Energy: 尚未提供此日期資料", flush=True)
        print("Optimality Gap: N/A", flush=True)
    else:
        print(f"Brute Force Energy: {result['brute_force_energy']:.6f}", flush=True)
        print(f"Optimality Gap: {result['optimality_gap'] * 100:.2f}%", flush=True)
    print("", flush=True)
    print(f"預期投組報酬: {result['expected_return'] * 100:.4f}%", flush=True)
    print(f"投組變異數: {result['risk']:.8f}", flush=True)
    print(f"供應鏈違規次數: {result['dependency_violations']}", flush=True)
    print("==================================================", flush=True)


def save_results(results, csv_path="qaoa_results_all_dates.csv", json_path="qaoa_results_all_dates.json"):
    fieldnames = [
        "date",
        "selected_tickers",
        "selected_count",
        "qaoa_qubo_fval",
        "original_energy",
        "brute_force_energy",
        "optimality_gap",
        "expected_return",
        "risk",
        "dependency_violations"
    ]

    csv_rows = []
    for result in results:
        row = result.copy()
        row["selected_tickers"] = ", ".join(row["selected_tickers"])
        csv_rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n五天 QAOA 結果已儲存為: {csv_path} / {json_path}", flush=True)


def print_summary_table(results):
    print("\n" + "=" * 100, flush=True)
    print("五個事件日 QAOA 求解結果總表", flush=True)
    print("=" * 100, flush=True)
    print(
        f"{'Date':<12} {'Count':>5} {'Energy':>12} {'Return':>12} {'Risk':>12}  Selected Tickers",
        flush=True
    )
    print("-" * 100, flush=True)
    for result in results:
        tickers = ", ".join(result["selected_tickers"])
        print(
            f"{result['date']:<12} "
            f"{result['selected_count']:>5} "
            f"{result['original_energy']:>12.6f} "
            f"{result['expected_return'] * 100:>11.4f}% "
            f"{result['risk']:>12.8f}  "
            f"{tickers}",
            flush=True
        )
    print("=" * 100, flush=True)


def main():
    print("=== 啟動 IBM Qiskit QAOA 量子求解引擎 ===\n", flush=True)

    mu_data = load_json_data("lstm_predicted_mu.json")
    sigma_data = load_json_data("sigma_matrices.json")

    results = []
    for target_date in TARGET_DATES:
        try:
            result = solve_for_date(target_date, mu_data, sigma_data)
        except ValueError as exc:
            print(f"輸入資料錯誤：{exc}", flush=True)
            continue

        results.append(result)
        print_result(result)

    if results:
        print_summary_table(results)
        save_results(results)

    print(
        "\n註：QUBO fval 含限制式懲罰項，論文比較建議使用「重新代回原始目標函數 Energy」與 Brute Force Energy 比較。",
        flush=True
    )


if __name__ == "__main__":
    main()