import json
import numpy as np
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

ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def build_chip4_qubo(mu, sigma, lmbda=0.5, p1=100, p2=100, p3=10):
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
    
    # 2. 持股數限制 (P1: K=5)
    qp.linear_constraint(linear={a: 1 for a in assets}, sense='==', rhs=5, name='Card_5')
    
    # 3. 市場配置限制 (P2)
    markets = {
        "US": (["NVDA", "AMD", "QCOM", "AMAT", "ASML"], 1),
        "TW": (["2330.TW", "2454.TW", "3711.TW", "6488.TWO"], 2),
        "JP": (["8035.T", "6857.T", "4063.T"], 1),
        "KR": (["005930.KS", "000660.KS", "042700.KS"], 1)
    }
    for mkt, (tickers, req) in markets.items():
        qp.linear_constraint(linear={t: 1 for t in tickers}, sense='==', rhs=req, name=f'Mkt_{mkt}')
        
    # 4. 供應鏈依賴懲罰 (P3)
    dependencies = [("NVDA", "2330.TW"), ("AMD", "2330.TW"), ("2330.TW", "ASML")]
    obj = qp.objective
    for dep, relies_on in dependencies:
        obj.linear[dep] += p3
        obj.quadratic[dep, relies_on] = obj.quadratic[dep, relies_on] - p3

    # 將 Quadratic Program 轉換為無限制的 QUBO
    conv = QuadraticProgramToQubo(penalty=p1) # 用極大值 P1 轉換等式限制
    qubo = conv.convert(qp)
    return qubo, assets

def main():
    print("=== 啟動 IBM Qiskit QAOA 量子求解引擎 ===\n")
    
    # 讀取資料
    mu_data = load_json_data("lstm_predicted_mu.json")
    sigma_data = load_json_data("sigma_matrices.json")
    
    # 我們先拿「事件衝擊日」來測試量子演算法
    target_date = "2026-04-08"
    print(f"👉 載入調倉日: {target_date} 之參數...")
    
    mu_dict = mu_data[target_date]
    sigma = sigma_data[target_date]
    mu = [mu_dict[ticker] for ticker in ASSETS]
    
    # 建立 QUBO 模型
    qubo_model, assets = build_chip4_qubo(mu, sigma, lmbda=0.5, p3=10)
    print(f"👉 QUBO 模型轉換完畢 (總變數含 Slack variables: {qubo_model.get_num_vars()})")
    
    # ==========================================
    # 核心：設定 QAOA 量子演算法
    # ==========================================
    print("\n👉 正在初始化 QAOA (p=1) 與古典優化器 COBYLA...")
    
    # p=1: 這是您論文設定的「NISQ 時代實務深度」
    # 優先使用 Aer 的 C++ 模擬後端；新版 Aer 需要先 transpile QAOA instruction
    if AerSimulator is not None:
        backend = AerSimulator(method="statevector")
        transpiler = generate_preset_pass_manager(optimization_level=1, backend=backend)
        qaoa = QAOA(
            sampler=Sampler(),
            optimizer=COBYLA(maxiter=200),
            reps=1,
            transpiler=transpiler
        )
    else:
        qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=200), reps=1)
    
    # 使用 MinimumEigenOptimizer 包裝 QAOA 來解 QUBO
    optimizer = MinimumEigenOptimizer(qaoa)
    
    print("⏳ QAOA 量子演算法運算中 (這會模擬量子態的演化，請稍候約 10~30 秒)...\n")
    result = optimizer.solve(qubo_model)
    
    # ==========================================
    # 解析結果
    # ==========================================
    print("==================================================")
    print("🏆 QAOA 求解完成！")
    
    # 找出 QAOA 選中的股票
    selected_indices = [i for i, val in enumerate(result.x[:15]) if val == 1.0]
    selected_tickers = [assets[i] for i in selected_indices]
    
    print(f"🎯 QAOA 找到之投資組合: {selected_tickers}")
    print(f"⚡ QAOA 算出的能量值 (fval): {result.fval:.6f}")
    
    # 帶入您剛剛算出的 Brute Force 最佳能量值來算 Optimality Gap
    # (請確認這個數字與您上一步 Brute Force 2026-04-08 的 Energy 一致)
    brute_force_energy = 0.002657 
    gap = abs(result.fval - brute_force_energy) / abs(brute_force_energy)
    
    print(f"📏 Optimality Gap (與全域最佳解差距): {gap * 100:.2f}%")
    print("==================================================")
    print("\n💡 註: 由於 QAOA 是啟發式演算法，如果能量值與 Brute Force 不同或跑出違規解，這正是我們要探討的「解品質落差」。")

if __name__ == "__main__":
    main()
