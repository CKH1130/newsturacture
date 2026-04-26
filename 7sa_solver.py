import json
import numpy as np
import dimod
import neal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
import warnings
warnings.filterwarnings('ignore')

ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def build_chip4_qubo(mu, sigma, lmbda=0.5, p1=100, p3=10):
    """與 QAOA 完全相同的 QUBO 建構邏輯，確保考卷一模一樣"""
    qp = QuadraticProgram()
    assets = ASSETS
    for asset in assets:
        qp.binary_var(name=asset)
        
    linear = {assets[i]: -(1 - lmbda) * mu[i] for i in range(len(assets))}
    quadratic = {}
    for i in range(len(assets)):
        for j in range(len(assets)):
            quadratic[(assets[i], assets[j])] = lmbda * sigma[i][j]
    qp.minimize(linear=linear, quadratic=quadratic)
    
    qp.linear_constraint(linear={a: 1 for a in assets}, sense='==', rhs=5)
    
    markets = {
        "US": (["NVDA", "AMD", "QCOM", "AMAT", "ASML"], 1),
        "TW": (["2330.TW", "2454.TW", "3711.TW", "6488.TWO"], 2),
        "JP": (["8035.T", "6857.T", "4063.T"], 1),
        "KR": (["005930.KS", "000660.KS", "042700.KS"], 1)
    }
    for mkt, (tickers, req) in markets.items():
        qp.linear_constraint(linear={t: 1 for t in tickers}, sense='==', rhs=req)
        
    dependencies = [("NVDA", "2330.TW"), ("AMD", "2330.TW"), ("2330.TW", "ASML")]
    obj = qp.objective
    for dep, relies_on in dependencies:
        obj.linear[dep] += p3
        obj.quadratic[dep, relies_on] = obj.quadratic[dep, relies_on] - p3

    conv = QuadraticProgramToQubo(penalty=p1)
    qubo = conv.convert(qp)
    return qubo, assets

def main():
    print("=== 啟動 D-Wave Simulated Annealing (SA) 古典求解引擎 ===\n")
    
    mu_data = load_json_data("lstm_predicted_mu.json")
    sigma_data = load_json_data("sigma_matrices.json")
    target_date = "2026-04-08"
    
    print(f"👉 載入調倉日: {target_date} 之參數...")
    mu = [mu_data[target_date][ticker] for ticker in ASSETS]
    sigma = sigma_data[target_date]
    
    qubo_model, assets = build_chip4_qubo(mu, sigma, lmbda=0.5, p3=10)
    
    # 將 Qiskit QUBO 轉換為 D-Wave 支援的 BQM (Binary Quadratic Model)
    linear_matrix = qubo_model.objective.linear.to_array()
    quadratic_matrix = qubo_model.objective.quadratic.to_array()
    offset = qubo_model.objective.constant
    
    linear_dict = {i: linear_matrix[i] for i in range(len(linear_matrix))}
    quadratic_dict = {(i, j): quadratic_matrix[i, j] for i in range(len(linear_matrix)) for j in range(len(linear_matrix)) if quadratic_matrix[i, j] != 0}
    
    bqm = dimod.BQM(linear_dict, quadratic_dict, offset, dimod.BINARY)
    
    print("⏳ SA 模擬退火演算法運算中 (Num Reads: 1000次)...\n")
    sampler = neal.SimulatedAnnealingSampler()
    
    # 執行 1000 次退火，取最好的一次
    sampleset = sampler.sample(bqm, num_reads=1000)
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    print("==================================================")
    print("🏆 SA 求解完成！")
    
    selected_tickers = [assets[i] for i in range(15) if best_sample[i] == 1]
    
    print(f"🎯 SA 找到之投資組合: {selected_tickers}")
    print(f"⚡ SA 算出的能量值 (fval): {best_energy:.6f}")
    
    brute_force_energy = 0.002657 
    gap = abs(best_energy - brute_force_energy) / abs(brute_force_energy)
    
    print(f"📏 Optimality Gap (與全域最佳解差距): {gap * 100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
    
