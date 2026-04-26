import json
import warnings
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV1 as Sampler

warnings.filterwarnings('ignore')

# ==========================================
# 🚨 請在這裡貼上您的 IBM Quantum API Token
# ==========================================
IBM_TOKEN = "YOUR_IBM_TOKEN_HERE"

ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

def build_qubo_model(mu, sigma):
    """建構統一的 QUBO 模型"""
    qp = QuadraticProgram()
    for a in ASSETS: qp.binary_var(name=a)
    
    linear = {ASSETS[i]: -0.5 * mu[i] for i in range(15)}
    quadratic = {(ASSETS[i], ASSETS[j]): 0.5 * sigma[i][j] for i in range(15) for j in range(15)}
    qp.minimize(linear=linear, quadratic=quadratic)
    
    qp.linear_constraint(linear={a: 1 for a in ASSETS}, sense='==', rhs=5)
    markets = {"US": 1, "TW": 2, "JP": 1, "KR": 1}
    mkt_assets = {
        "US": ["NVDA", "AMD", "QCOM", "AMAT", "ASML"],
        "TW": ["2330.TW", "2454.TW", "3711.TW", "6488.TWO"],
        "JP": ["8035.T", "6857.T", "4063.T"],
        "KR": ["005930.KS", "000660.KS", "042700.KS"]
    }
    for mkt, req in markets.items():
        qp.linear_constraint(linear={t: 1 for t in mkt_assets[mkt]}, sense='==', rhs=req)
        
    obj = qp.objective
    for dep, relies_on in [("NVDA", "2330.TW"), ("AMD", "2330.TW"), ("2330.TW", "ASML")]:
        obj.linear[dep] += 10
        coeff = obj.quadratic[dep, relies_on] if (dep, relies_on) in obj.quadratic.coefficients else 0
        obj.quadratic[dep, relies_on] = coeff - 10

    return QuadraticProgramToQubo(penalty=100).convert(qp)

def main():
    print("=== 🚀 連線至 IBM Quantum 真實量子硬體 ===\n")
    
    # 登入 IBM 伺服器
    print("🔑 正在驗證 IBM Token...")
    QiskitRuntimeService.save_account(channel="ibm_quantum", token=IBM_TOKEN, set_as_default=True, overwrite=True)
    service = QiskitRuntimeService()
    
    # 選擇目前最空閒的免費真機 (通常是 ibmq_qasm_simulator 或 ibm_brisbane/ibm_kyoto)
    # 這裡我們讓系統自動挑選排隊最少的最少 127 qubits 真機
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=15)
    print(f"✅ 成功連線！已鎖定真實量子電腦: {backend.name} (Qubits: {backend.num_qubits})\n")

    with open("lstm_predicted_mu.json", 'r') as f: mu_data = json.load(f)
    with open("sigma_matrices.json", 'r') as f: sigma_data = json.load(f)
    
    # 論文鎖定的兩大極端日
    target_dates = ["2019-05-13", "2026-04-08"]

    for date in target_dates:
        print(f"==========================================")
        print(f"📅 準備發送調倉日: {date} 之任務")
        mu = [mu_data[date][k] for k in ASSETS]
        sigma = sigma_data[date]
        qubo_model = build_qubo_model(mu, sigma)
        
        # 開啟 Runtime Session 保持連線
        print(f"📡 正在將電路送往 {backend.name} 佇列排隊 (請耐心等候，依伺服器狀況可能需要數十分鐘至數小時)...")
        with Session(service=service, backend=backend) as session:
            # 呼叫真機上的 Sampler
            sampler = Sampler(session=session)
            optimizer = MinimumEigenOptimizer(QAOA(sampler=sampler, optimizer=COBYLA(maxiter=50), reps=1))
            
            try:
                result = optimizer.solve(qubo_model)
                qaoa_selected = [ASSETS[i] for i, val in enumerate(result.x[:15]) if val == 1.0]
                
                print(f"🏆 【真機結果回傳】調倉日: {date}")
                print(f"🎯 真機選定投資組合: {qaoa_selected}")
                print(f"⚡ 真機算出的能量值: {result.fval:.6f}")
                print(f"💡 (可與表一之 QAOA 模擬能量比較，觀察雜訊造成的退化)")
                
            except Exception as e:
                print(f"❌ 任務執行失敗，原因: {e}")
        print(f"==========================================\n")

if __name__ == "__main__":
    main()