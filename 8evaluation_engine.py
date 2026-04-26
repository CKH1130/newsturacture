import json
import numpy as np
import itertools
import pandas as pd
import time
import dimod
import neal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except ImportError:
    AerSimulator = None
    generate_preset_pass_manager = None
    try:
        from qiskit.primitives import StatevectorSampler as Sampler
    except ImportError:
        from qiskit.primitives import Sampler
import warnings
warnings.filterwarnings('ignore')

# === 常數與設定 ===
ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]
MARKETS = {
    "US": ["NVDA", "AMD", "QCOM", "AMAT", "ASML"],
    "TW": ["2330.TW", "2454.TW", "3711.TW", "6488.TWO"],
    "JP": ["8035.T", "6857.T", "4063.T"],
    "KR": ["005930.KS", "000660.KS", "042700.KS"]
}
DEPENDENCIES = [("NVDA", "2330.TW"), ("AMD", "2330.TW"), ("2330.TW", "ASML")]

def check_violations(selected_tickers):
    """裁判函式：檢查名單的違規次數"""
    violations = 0
    # 1. 檢查市場配額 (TW:2, US:1, JP:1, KR:1)
    us_c = sum(1 for t in selected_tickers if t in MARKETS["US"])
    tw_c = sum(1 for t in selected_tickers if t in MARKETS["TW"])
    jp_c = sum(1 for t in selected_tickers if t in MARKETS["JP"])
    kr_c = sum(1 for t in selected_tickers if t in MARKETS["KR"])
    violations += abs(us_c - 1) + abs(tw_c - 2) + abs(jp_c - 1) + abs(kr_c - 1)
    
    # 2. 檢查供應鏈
    if 'NVDA' in selected_tickers and '2330.TW' not in selected_tickers: violations += 1
    if 'AMD' in selected_tickers and '2330.TW' not in selected_tickers: violations += 1
    if '2330.TW' in selected_tickers and 'ASML' not in selected_tickers: violations += 1
    
    return violations

def build_qubo_model(mu, sigma, lmbda=0.5, p1=100, p3=10):
    """建構統一的 QUBO 模型"""
    qp = QuadraticProgram()
    for a in ASSETS: qp.binary_var(name=a)
    
    linear = {ASSETS[i]: -(1 - lmbda) * mu[i] for i in range(15)}
    quadratic = {(ASSETS[i], ASSETS[j]): lmbda * sigma[i][j] for i in range(15) for j in range(15)}
    qp.minimize(linear=linear, quadratic=quadratic)
    
    qp.linear_constraint(linear={a: 1 for a in ASSETS}, sense='==', rhs=5)
    reqs = {"US": 1, "TW": 2, "JP": 1, "KR": 1}
    for mkt, req in reqs.items():
        qp.linear_constraint(linear={t: 1 for t in MARKETS[mkt]}, sense='==', rhs=req)
        
    obj = qp.objective
    for dep, relies_on in DEPENDENCIES:
        obj.linear[dep] += p3
        obj.quadratic[dep, relies_on] = obj.quadratic[dep, relies_on] - p3

    return QuadraticProgramToQubo(penalty=p1).convert(qp)

def build_qaoa_optimizer(maxiter=200, callback=None):
    if AerSimulator is not None:
        backend = AerSimulator(method="statevector")
        transpiler = generate_preset_pass_manager(optimization_level=1, backend=backend)
        qaoa = QAOA(
            sampler=Sampler(),
            optimizer=COBYLA(maxiter=maxiter),
            reps=1,
            callback=callback,
            transpiler=transpiler
        )
    else:
        qaoa = QAOA(
            sampler=Sampler(),
            optimizer=COBYLA(maxiter=maxiter),
            reps=1,
            callback=callback
        )
    return MinimumEigenOptimizer(qaoa)

def format_markdown_table(headers, rows):
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
    return lines

def main():
    print("=== 🚀 論文裁判引擎啟動 (含 QAOA 模擬計時版) ===\n")
    print("⏳ 正在為 5 個代表日執行 [Brute Force, SA, QAOA] 解品質比較，並針對 QAOA 模擬器計時...\n")
    
    with open("lstm_predicted_mu.json", 'r') as f: mu_data = json.load(f)
    with open("sigma_matrices.json", 'r') as f: sigma_data = json.load(f)
    
    dates = ["2019-05-13", "2019-06-19", "2019-06-24", "2020-03-04", "2026-04-08"]
    
    # 用來存表格資料
    results_table = []
    
    for date in dates:
        print(f"👉 正在評估: {date}...")
        mu = [mu_data[date][k] for k in ASSETS]
        sigma = sigma_data[date]
        qubo_model = build_qubo_model(mu, sigma)
        
        # --- 1. Brute Force ---
        bf_energy = float('inf')
        bf_violations = 0
        for combo in itertools.combinations(range(15), 5):
            x = np.zeros(15); x[list(combo)] = 1
            selected = [ASSETS[i] for i in combo]
            v_count = check_violations(selected)
            if v_count > 0: continue # BF 只看完美解
            
            risk = np.dot(x.T, np.dot(sigma, x))
            ret = np.dot(mu, x)
            energy = 0.5 * risk - 0.5 * ret
            if energy < bf_energy: bf_energy = energy
            
        # --- 2. SA (Simulated Annealing) ---
        lin_mat = qubo_model.objective.linear.to_array()
        quad_mat = qubo_model.objective.quadratic.to_array()
        offset = qubo_model.objective.constant
        bqm = dimod.BQM({i: lin_mat[i] for i in range(15)}, {(i,j): quad_mat[i,j] for i in range(15) for j in range(15) if quad_mat[i,j]!=0}, offset, dimod.BINARY)
        sa_samples = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=1000)
        sa_selected = [ASSETS[i] for i in range(15) if sa_samples.first.sample[i] == 1]
        sa_energy = sa_samples.first.energy
        sa_viol = check_violations(sa_selected)
        sa_gap = abs(sa_energy - bf_energy) / abs(bf_energy)
        
        # --- 3. QAOA ---
        qaoa_eval_history = []

        def qaoa_callback(eval_count, parameters, mean, metadata):
            qaoa_eval_history.append(eval_count)

        optimizer = build_qaoa_optimizer(maxiter=200, callback=qaoa_callback)
        start_qaoa = time.perf_counter()
        qaoa_res = optimizer.solve(qubo_model)
        qaoa_duration = time.perf_counter() - start_qaoa
        qaoa_selected = [ASSETS[i] for i, val in enumerate(qaoa_res.x[:15]) if val == 1.0]
        qaoa_energy = qaoa_res.fval
        qaoa_viol = check_violations(qaoa_selected)
        qaoa_gap = abs(qaoa_energy - bf_energy) / abs(bf_energy)
        qaoa_evals = max(qaoa_eval_history) if qaoa_eval_history else len(qaoa_eval_history)
        
        results_table.append({
            "Date": date,
            "BF_Energy": bf_energy,
            "SA_Gap": f"{sa_gap*100:.2f}%", "SA_Viol": sa_viol,
            "QAOA_Gap": f"{qaoa_gap*100:.2f}%", "QAOA_Viol": qaoa_viol,
            "QAOA_Time": qaoa_duration,
            "QAOA_Evals": qaoa_evals
        })

    # === 印出 Markdown 表格 ===
    table_headers = ["Date", "BF Energy", "SA Gap", "SA Viol", "QAOA Gap", "QAOA Viol"]
    table_rows = [
        [
            r["Date"],
            f"{r['BF_Energy']:.6f}",
            r["SA_Gap"],
            str(r["SA_Viol"]),
            r["QAOA_Gap"],
            str(r["QAOA_Viol"]),
        ]
        for r in results_table
    ]
    print("\n\n" + "="*70)
    print("📋 [論文可以直接複製貼上] 表一：各求解器解品質與約束滿足度比較")
    print("="*70)
    for line in format_markdown_table(table_headers, table_rows):
        print(line)
    print("="*70)
    print("欄位說明：BF Energy = Brute Force 基準能量；Viol = 違規次數。")
    print("💡 註解：違規次數包含『市場配置』與『供應鏈依賴』之違反總和。")

    # === 印出 QAOA 模擬器 vs 真機效率比較表 ===
    efficiency_headers = ["Date", "Sim Time", "Sim Evals", "HW Usage", "HW Job ID"]
    efficiency_rows = [
        [
            r["Date"],
            f"{r['QAOA_Time']:.4f}s",
            str(r["QAOA_Evals"]),
            "TBD",
            "TBD",
        ]
        for r in results_table
    ]
    avg_qaoa_time = sum(r["QAOA_Time"] for r in results_table) / len(results_table)

    print("\n" + "="*70)
    print("📋 表二：QAOA 模擬器與真機執行效率比較")
    print("="*70)
    for line in format_markdown_table(efficiency_headers, efficiency_rows):
        print(line)
    print("="*70)
    print(f"QAOA 模擬平均耗時：{avg_qaoa_time:.4f}s")
    print("💡 註：Sim Time 為古典 CPU 模擬量子運算之時間；HW Usage/Job ID 待 9ibm_real_hardware.py 真機回填。")

    quality_output = "evaluation_quality_table.csv"
    efficiency_output = "evaluation_efficiency_table.csv"
    pd.DataFrame([
        {
            "Date": r["Date"],
            "BF_Energy": r["BF_Energy"],
            "SA_Gap": r["SA_Gap"],
            "SA_Viol": r["SA_Viol"],
            "QAOA_Gap": r["QAOA_Gap"],
            "QAOA_Viol": r["QAOA_Viol"],
        }
        for r in results_table
    ]).to_csv(quality_output, index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {
            "Date": r["Date"],
            "QAOA_Sim_Time_Sec": r["QAOA_Time"],
            "QAOA_Function_Evals": r["QAOA_Evals"],
            "IBM_HW_Usage_Sec": "",
            "IBM_Job_ID": "",
        }
        for r in results_table
    ]).to_csv(efficiency_output, index=False, encoding="utf-8-sig")
    print(f"已輸出：{quality_output}")
    print(f"已輸出：{efficiency_output}")

if __name__ == "__main__":
    main()
