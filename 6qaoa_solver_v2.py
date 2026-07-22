import json
import itertools
import numpy as np
import time 
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except ImportError:
    AerSimulator = None
    AerSampler = None
    NoiseModel = None
    ReadoutError = None
    depolarizing_error = None
    generate_preset_pass_manager = None

try:
    from qiskit.primitives import StatevectorSampler
except ImportError:
    StatevectorSampler = None

import warnings
warnings.filterwarnings('ignore')

algorithm_globals.random_seed = 42

ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]
QAOA_RESULTS_FILE = "qaoa_results.json"
QAOA_MAXITER = 200
QAOA_SHOTS = 1024
NOISE_1Q_ERROR = 0.001
NOISE_2Q_ERROR = 0.01
NOISE_READOUT_P01 = 0.02
NOISE_READOUT_P10 = 0.03

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def build_chip4_qubo(mu, sigma, lmbda=0.5, p1=100, p3=10):
    """將參數轉換為 Qiskit 的 QuadraticProgram"""
    qp = QuadraticProgram(name="Chip4_QAOA")
    
    assets = ASSETS
    
    # 宣告二元變數
    for asset in assets:
        qp.binary_var(name=asset)
        
    # 1. 基礎目標函數 (H_rr)
    linear = {assets[i]: -(1 - lmbda) * mu[i] for i in range(len(assets))}
    quadratic = {}
    for i in range(len(assets)):
        for j in range(len(assets)):
            quadratic[(assets[i], assets[j])] = lmbda * sigma[i][j]
    qp.minimize(linear=linear, quadratic=quadratic)
    
    # 2. 持股數限制 (P1: K=4)
    qp.linear_constraint(linear={a: 1 for a in assets}, sense='==', rhs=4, name='Card_4')
    
    # 3. 市場配置限制 (P2)
    markets = {
        "US": (["NVDA", "AMD", "QCOM", "AMAT", "ASML"], 1),
        "TW": (["2330.TW", "2454.TW", "3711.TW", "6488.TWO"], 1),
        "JP": (["8035.T", "6857.T", "4063.T"], 1),
        "KR": (["005930.KS", "000660.KS", "042700.KS"], 1)
    }
    for mkt, (tickers, req) in markets.items():
        qp.linear_constraint(linear={t: 1 for t in tickers}, sense='==', rhs=req, name=f'Mkt_{mkt}')
        
    # 4. 供應鏈依賴懲罰 (P3)
    dependencies = [
        ("NVDA", "2330.TW"),          # X_NVIDIA(1 - X_TSMC)
        ("AMD", "2330.TW"),           # X_AMD(1 - X_TSMC)
        ("QCOM", "2330.TW"),          # X_Qualcomm(1 - X_TSMC)
        ("2330.TW", "ASML"),          # X_TSMC(1 - X_ASML)
        ("2330.TW", "6488.TWO"),      # X_TSMC(1 - X_GlobalWafers) [環球晶]
        ("005930.KS", "2330.TW")      # X_Samsung(1 - X_TSMC) [三星]
    ]
    obj = qp.objective
    for dep, relies_on in dependencies:
        obj.linear[dep] += p3
        obj.quadratic[dep, relies_on] = obj.quadratic[dep, relies_on] - p3

    # 將 Quadratic Program 轉換為無限制的 QUBO
    conv = QuadraticProgramToQubo(penalty=p1)  # 用極大值 P1 轉換等式限制
    qubo = conv.convert(qp)
    return qubo, assets

def compute_brute_force_energy(mu, sigma, lmbda=0.5, p3=10):
    markets = {
        "US": ["NVDA", "AMD", "QCOM", "AMAT", "ASML"],
        "TW": ["2330.TW", "2454.TW", "3711.TW", "6488.TWO"],
        "JP": ["8035.T", "6857.T", "4063.T"],
        "KR": ["005930.KS", "000660.KS", "042700.KS"],
    }
    dependencies = [
        ("NVDA", "2330.TW"),          # X_NVIDIA(1 - X_TSMC)
        ("AMD", "2330.TW"),           # X_AMD(1 - X_TSMC)
        ("QCOM", "2330.TW"),          # X_Qualcomm(1 - X_TSMC)
        ("2330.TW", "ASML"),          # X_TSMC(1 - X_ASML)
        ("2330.TW", "6488.TWO"),      # X_TSMC(1 - X_GlobalWafers) [環球晶]
        ("005930.KS", "2330.TW")      # X_Samsung(1 - X_TSMC) [三星]
    ]

    best_energy = float("inf")
    for combo in itertools.combinations(range(len(ASSETS)), 4):
        x = np.zeros(len(ASSETS))
        x[list(combo)] = 1
        selected = [ASSETS[i] for i in combo]

        if not (
            sum(t in markets["US"] for t in selected) == 1
            and sum(t in markets["TW"] for t in selected) == 1
            and sum(t in markets["JP"] for t in selected) == 1
            and sum(t in markets["KR"] for t in selected) == 1
        ):
            continue

        risk = np.dot(x.T, np.dot(sigma, x))
        return_val = np.dot(mu, x)
        violations = sum(dep in selected and relies_on not in selected for dep, relies_on in dependencies)
        energy = lmbda * risk - (1 - lmbda) * return_val + p3 * violations
        if energy < best_energy:
            best_energy = energy

    return best_energy

def validate_portfolio(selected_tickers):
    markets = {
        "US": ["NVDA", "AMD", "QCOM", "AMAT", "ASML"],
        "TW": ["2330.TW", "2454.TW", "3711.TW", "6488.TWO"],
        "JP": ["8035.T", "6857.T", "4063.T"],
        "KR": ["005930.KS", "000660.KS", "042700.KS"],
    }
    required = {"US": 1, "TW": 1, "JP": 1, "KR": 1}
    violations = 0

    if len(selected_tickers) != 4:
        violations += abs(len(selected_tickers) - 4)

    for mkt, tickers in markets.items():
        count = sum(ticker in tickers for ticker in selected_tickers)
        violations += abs(count - required[mkt])

    dependencies = [
        ("NVDA", "2330.TW"),          # X_NVIDIA(1 - X_TSMC)
        ("AMD", "2330.TW"),           # X_AMD(1 - X_TSMC)
        ("QCOM", "2330.TW"),          # X_Qualcomm(1 - X_TSMC)
        ("2330.TW", "ASML"),          # X_TSMC(1 - X_ASML)
        ("2330.TW", "6488.TWO"),      # X_TSMC(1 - X_GlobalWafers) [環球晶]
        ("005930.KS", "2330.TW")      # X_Samsung(1 - X_TSMC) [三星]
    ]
    violations += sum(
        1 for dep, relies_on in dependencies
        if dep in selected_tickers and relies_on not in selected_tickers
    )

    return violations

def portfolio_energy(selected_tickers, mu, sigma, lmbda=0.5):
    x = np.array([1.0 if ticker in selected_tickers else 0.0 for ticker in ASSETS])
    risk = np.dot(x.T, np.dot(sigma, x))
    return_val = np.dot(mu, x)
    return lmbda * risk - (1 - lmbda) * return_val

def decode_selected(sample_x, assets):
    return [
        assets[i]
        for i, value in enumerate(sample_x[:len(assets)])
        if value > 0.5
    ]

def choose_best_qaoa_portfolio(result, assets, mu, sigma):
    feasible_candidates = []
    all_candidates = []

    for sample in result.samples:
        selected = decode_selected(sample.x, assets)
        violations = validate_portfolio(selected)
        energy = portfolio_energy(selected, mu, sigma)
        candidate = {
            "selected": selected,
            "energy": float(energy),
            "qubo_fval": float(sample.fval),
            "probability": float(sample.probability),
            "violations": int(violations),
        }
        all_candidates.append(candidate)
        if violations == 0:
            feasible_candidates.append(candidate)

    if feasible_candidates:
        best = min(feasible_candidates, key=lambda item: item["energy"])
        best["source"] = "qaoa_feasible_sample"
        return best

    projected = project_to_feasible_portfolio(result.x, assets, mu, sigma)
    projected["source"] = "projected_from_qaoa"
    return projected

def project_to_feasible_portfolio(sample_x, assets, mu, sigma):
    raw_bits = np.array([1.0 if value > 0.5 else 0.0 for value in sample_x[:len(assets)]])
    best = None

    for combo in itertools.combinations(range(len(assets)), 4):
        selected = [assets[i] for i in combo]
        if validate_portfolio(selected) != 0:
            continue

        bits = np.zeros(len(assets))
        bits[list(combo)] = 1.0
        distance = int(np.sum(np.abs(bits - raw_bits)))
        energy = portfolio_energy(selected, mu, sigma)

        candidate = {
            "selected": selected,
            "energy": float(energy),
            "qubo_fval": None,
            "probability": None,
            "violations": 0,
            "hamming_distance": distance,
        }

        if best is None or (distance, energy) < (best["hamming_distance"], best["energy"]):
            best = candidate

    if best is None:
        selected = decode_selected(sample_x, assets)
        return {
            "selected": selected,
            "energy": float("inf"),
            "qubo_fval": float("inf"),
            "probability": None,
            "violations": int(validate_portfolio(selected)),
            "hamming_distance": None,
        }

    return best

def raw_qaoa_choice(result, assets):
    selected = decode_selected(result.x, assets)
    return {
        "selected": selected,
        "energy": float(result.fval),
        "qubo_fval": float(result.fval),
        "probability": None,
        "violations": int(validate_portfolio(selected)),
    }


def build_noise_model():
    if NoiseModel is None or depolarizing_error is None or ReadoutError is None:
        raise RuntimeError("目前無法建立雜訊模型，請確認 qiskit-aer noise 模組可用。")

    noise_model = NoiseModel()
    one_qubit_error = depolarizing_error(NOISE_1Q_ERROR, 1)
    two_qubit_error = depolarizing_error(NOISE_2Q_ERROR, 2)
    readout_error = ReadoutError([
        [1 - NOISE_READOUT_P01, NOISE_READOUT_P01],
        [NOISE_READOUT_P10, 1 - NOISE_READOUT_P10],
    ])

    noise_model.add_all_qubit_quantum_error(
        one_qubit_error,
        ["x", "sx", "h", "rx", "ry", "rz"],
    )
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model

def build_qaoa_optimizer(mode="ideal"):
    transpiler = None
    if AerSimulator is not None and AerSampler is not None and generate_preset_pass_manager is not None:
        if mode == "noisy":
            print("   [系統] 啟動 Aer depolarizing/readout 雜訊模擬。", flush=True)
            noise_model = build_noise_model()
            backend = AerSimulator(noise_model=noise_model)
            sampler_options = {"backend_options": {"noise_model": noise_model}}
        else:
            print("   [系統] 啟動 Aer statevector 無雜訊模擬。", flush=True)
            backend = AerSimulator(method="statevector")
            sampler_options = {"backend_options": {"method": "statevector"}}

        transpiler = generate_preset_pass_manager(optimization_level=1, backend=backend)
        sampler = AerSampler(
            seed=42,
            default_shots=QAOA_SHOTS,
            options=sampler_options,
        )
    elif StatevectorSampler is not None:
        if mode == "noisy":
            raise RuntimeError("目前缺少 qiskit-aer，無法執行 noisy 模擬。")
        print("   [系統] 啟動 Qiskit StatevectorSampler 無雜訊 fallback。", flush=True)
        sampler = StatevectorSampler(seed=42)
    else:
        raise RuntimeError("目前無法啟動無雜訊模擬，請安裝 qiskit-aer 或確認 qiskit.primitives.StatevectorSampler 可用。")

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=QAOA_MAXITER),
        reps=1,
        transpiler=transpiler,
    )

    return MinimumEigenOptimizer(qaoa)

def solve_one_date(record, mu_data, sigma_data):
    target_date = record["date"]
    category = record["category"]
    print("==================================================", flush=True)
    print(f"📅 {category}: {target_date}", flush=True)

    if target_date not in mu_data:
        raise RuntimeError(f"lstm_predicted_mu.json 缺少 {target_date}。請重新執行 3lstm_predictor.py。")
    if target_date not in sigma_data:
        raise RuntimeError(f"sigma_matrices.json 缺少 {target_date}。請重新執行 4covariance_extractor.py。")

    mu_dict = mu_data[target_date]
    sigma = sigma_data[target_date]
    mu = [mu_dict[ticker] for ticker in ASSETS]

    qubo_model, assets = build_chip4_qubo(mu, sigma, lmbda=0.5, p3=10)
    print(f"👉 QUBO 模型轉換完畢 (總變數含 Slack variables: {qubo_model.get_num_vars()})", flush=True)
    print("⏳ QAOA 量子演算法運算中...\n", flush=True)

    brute_force_energy = compute_brute_force_energy(np.array(mu), np.array(sigma))
    results_dict = {
        "Date": target_date,
        "Category": category,
        "BF_Energy": float(brute_force_energy),
    }

    for mode, label in [("ideal", "Ideal"), ("noisy", "Noisy")]:
        mode_start_time = time.time()  
        
        optimizer = build_qaoa_optimizer(mode=mode)
        result = optimizer.solve(qubo_model)
        qaoa_choice = choose_best_qaoa_portfolio(result, assets, np.array(mu), np.array(sigma))
        
        mode_elapsed_time = time.time() - mode_start_time
        
        selected_tickers = qaoa_choice["selected"]
        
        # 加上絕對值保護，避免除以零或符號錯誤
        gap = abs(qaoa_choice["energy"] - brute_force_energy) / abs(brute_force_energy) if brute_force_energy != 0 else 0
        violations = qaoa_choice["violations"]

        results_dict[f"{label}_Energy"] = float(qaoa_choice["energy"])
        results_dict[f"{label}_QUBO_Fval"] = qaoa_choice["qubo_fval"]
        results_dict[f"{label}_Gap"] = float(gap)
        results_dict[f"{label}_Selected"] = selected_tickers
        results_dict[f"{label}_Violations"] = int(violations)
        results_dict[f"{label}_Probability"] = qaoa_choice["probability"]
        results_dict[f"{label}_Source"] = qaoa_choice["source"]
        results_dict[f"{label}_HammingDistance"] = qaoa_choice.get("hamming_distance")
        results_dict[f"{label}_Time"] = float(mode_elapsed_time)

        print(f"🏆 [{label}] QAOA 求解完成！ (耗時: {mode_elapsed_time:.2f} 秒)", flush=True)
        print(f"Gap: {gap * 100:.2f}%", flush=True)
        print(f"⚡ 原始目標能量值: {qaoa_choice['energy']:.6f}", flush=True)
        qubo_fval = qaoa_choice["qubo_fval"]
        qubo_fval_text = f"{qubo_fval:.6f}" if qubo_fval is not None else "n/a"
        print(f"🧮 QUBO fval: {qubo_fval_text}", flush=True)
        print(f"🔎 解碼來源: {qaoa_choice['source']}", flush=True)
        print(f"⚠️ 違規次數: {violations}", flush=True)
        print(f"🎯 投資組合: {selected_tickers}", flush=True)

    print("==================================================\n", flush=True)
    return results_dict

def format_gap(gap):
    return f"{gap * 100:.2f}%" if gap is not None else "skipped"

def format_value(value):
    return str(value) if value is not None else "skipped"

def main():
    total_start_time = time.time()
    
    print("=== 啟動 IBM Qiskit QAOA 量子求解引擎 ===\n", flush=True)
    print("設定：K=4，市場配置 (US, TW, JP, KR) = (1, 1, 1, 1)。", flush=True)
    print(
        "引擎：Ideal 無雜訊 + Noisy depolarizing/readout Aer 模擬。\n",
        flush=True,
    )
    
    # 讀取數據 (預期報酬與共變異數)
    mu_data = load_json_data("lstm_predicted_mu.json")
    sigma_data = load_json_data("sigma_matrices.json")
    
    # 🚨 寫死論文的 5 個代表日，不再依賴外部檔案
    representative_records = [
        {"category": "極端大跌日", "date": "2019-05-13"},
        {"category": "極端大漲日", "date": "2019-06-19"},
        {"category": "平穩日",   "date": "2019-06-24"},
        {"category": "高波動日", "date": "2020-03-04"},
        {"category": "事件衝擊日", "date": "2026-04-27"}
    ]
    
    print(f"👉 已載入 {len(representative_records)} 個代表日，開始逐日求解...\n", flush=True)

    results = [
        solve_one_date(record, mu_data, sigma_data)
        for record in representative_records
    ]

    with open(QAOA_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4, allow_nan=False)

    print("📋 QAOA 五大代表日摘要", flush=True)
    for result in results:
        print(
            f"- {result['Date']} / {result['Category']}: "
            f"BF {result['BF_Energy']:.6f}, "
            f"Ideal Gap {format_gap(result['Ideal_Gap'])}, "
            f"Noisy Gap {format_gap(result['Noisy_Gap'])}, "
            f"Ideal Viol {format_value(result['Ideal_Violations'])}, "
            f"Noisy Viol {format_value(result['Noisy_Violations'])}",
            flush=True
        )
        print(
            f"    Ideal Selected: {result['Ideal_Selected']} (耗時: {result['Ideal_Time']:.2f}s)",
            flush=True
        )
        print(
            f"    Noisy Selected: {result['Noisy_Selected']} (耗時: {result['Noisy_Time']:.2f}s)",
            flush=True
        )
    print(f"\n已輸出完整 QAOA 結果：{QAOA_RESULTS_FILE}", flush=True)
    print("\n💡 註: 由於 QAOA 是啟發式演算法，如果能量值與 Brute Force 不同或跑出違規解，這正是我們要探討的「解品質落差」。", flush=True)
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\n⏱️ 程式總執行時間: {total_elapsed_time:.2f} 秒", flush=True)

if __name__ == "__main__":
    main()