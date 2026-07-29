import json
import warnings
import numpy as np
import time

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz

warnings.filterwarnings('ignore')

# ==========================================
# 🚨 API Token 設定
# ==========================================
IBM_TOKEN = "fKDFFQqzS4QK09QvZEdRVZjP0QHRfAIBDCseLJzoewqs"

ASSETS = [
    "NVDA", "AMD", "QCOM", "AMAT", "ASML",
    "2330.TW", "2454.TW", "3711.TW", "6488.TWO",
    "8035.T", "6857.T", "4063.T",
    "005930.KS", "000660.KS", "042700.KS"
]

def build_qubo_model(mu, sigma):
    """建構包含供應鏈相依性的 QUBO (二次無限制二元最佳化) 模型"""
    
    qp = QuadraticProgram()
    for a in ASSETS: qp.binary_var(name=a)
    
    linear = {ASSETS[i]: -0.5 * mu[i] for i in range(15)}
    quadratic = {(ASSETS[i], ASSETS[j]): 0.5 * sigma[i][j] for i in range(15) for j in range(15)}
    qp.minimize(linear=linear, quadratic=quadratic)
    
    qp.linear_constraint(linear={a: 1 for a in ASSETS}, sense='==', rhs=4)
    mkt_assets = {
        "US": ["NVDA", "AMD", "QCOM", "AMAT", "ASML"],
        "TW": ["2330.TW", "2454.TW", "3711.TW", "6488.TWO"],
        "JP": ["8035.T", "6857.T", "4063.T"],
        "KR": ["005930.KS", "000660.KS", "042700.KS"]
    }
    
    markets = {"US": 1, "TW": 1, "JP": 1, "KR": 1}
    for mkt, req in markets.items():
        qp.linear_constraint(linear={t: 1 for t in mkt_assets[mkt]}, sense='==', rhs=req)
        
    obj = qp.objective
    for dep, relies_on in [("NVDA", "2330.TW"),("AMD", "2330.TW"),("QCOM", "2330.TW"),("2330.TW", "ASML"),("2330.TW", "6488.TWO"),("005930.KS", "2330.TW") ]:
        obj.linear[dep] += 10
        obj.quadratic[dep, relies_on] -= 10

    return QuadraticProgramToQubo(penalty=100).convert(qp)

def calculate_original_energy(selected_tickers, mu, sigma, lmbda=0.5):
    """將真機選出的股票代回原始目標函數，計算純粹的財務 Energy (不含 QUBO 懲罰項)"""
    
    x = np.zeros(len(ASSETS))
    for ticker in selected_tickers:
        if ticker in ASSETS:
            idx = ASSETS.index(ticker)
            x[idx] = 1

    risk = float(x.T @ sigma @ x)
    expected_return = float(np.dot(mu, x))
    
    return lmbda * risk - (1 - lmbda) * expected_return

def main():
    total_start_time = time.time() 
    
    print("=== 🚀 連線至 IBM Quantum 真實量子電腦===\n")
    
    print("🔑 正在驗證 IBM Token...")
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=IBM_TOKEN, set_as_default=True, overwrite=True)
    service = QiskitRuntimeService()
    
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=15)
    
    sampler = Sampler(mode=backend)
    print(f"✅ 成功連線！已鎖定真實量子電腦: {backend.name} (Qubits: {backend.num_qubits})\n")
    try:
        with open("lstm_predicted_mu.json", 'r') as f: mu_data = json.load(f)
        with open("sigma_matrices.json", 'r') as f: sigma_data = json.load(f)
    except FileNotFoundError:
        print("❌ 錯誤：找不到數據檔案 (json)，請確認檔案路徑。")
        return
        
    target_dates = ["2019-05-13","2019-06-19","2019-06-24","2020-03-04","2026-04-27"]

    for date in target_dates:
        date_start_time = time.time()
        
        print(f"==========================================")
        print(f"📅 準備發送調倉日: {date} 之任務")
        
        if date not in mu_data: continue

        mu = [mu_data[date][k] for k in ASSETS]
        sigma = np.array(sigma_data[date])
        
        qubo_model = build_qubo_model(mu, sigma)
        observable, offset = qubo_model.to_ising() 
        
        ansatz = QAOAAnsatz(cost_operator=observable, reps=1)
        ansatz.measure_all() 
        
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_circuit = pm.run(ansatz)

        try:
            print(f"📡 正在發送 ISA 指令至 {backend.name} (15 Qubits)...")
            
            initial_theta = [0.1] * ansatz.num_parameters
            
            job_start_time = time.time()
            
            job = sampler.run([(isa_circuit, initial_theta)])
            print(f"🆔 Job ID: {job.job_id()}")
            
            # --- 監控排隊與執行狀態 ---
            print("⏳ 任務已進入 IBM 系統，正在監控進度...")
            while True:
                status = job.status()
                status_name = status.name if hasattr(status, 'name') else str(status)
                
                if status_name == 'DONE':
                    print("\n✅ 任務執行完成！正在抓取結果數據...")
                    break
                elif status_name in ['FAILED', 'CANCELLED', 'ERROR']:
                    raise RuntimeError(f"IBM 任務異常終止，狀態: {status_name}")
                else:
                    print(f"   [目前狀態]: {status_name} (請耐心等待...)", end='\r')
                    time.sleep(30)
            
            job_elapsed_time = time.time() - job_start_time 
            
            result = job.result() 
            pub_result = result[0]
            
            data_bin = pub_result.data
            counts = None
            

            for field in data_bin:
                field_data = getattr(data_bin, field)
                if hasattr(field_data, 'get_counts'):
                    counts = field_data.get_counts()
                    break
            
            if counts is None:
                raise ValueError("無法從 DataBin 中找到有效的測量計數數據")

            best_bitstring = max(counts, key=counts.get)
            
            x_sample = [int(bit) for bit in best_bitstring[::-1]]
            
            fval = qubo_model.objective.evaluate(x_sample)
            
            qaoa_selected = [ASSETS[i] for i, val in enumerate(x_sample) if val == 1]
            
            real_energy = calculate_original_energy(qaoa_selected, mu, sigma)
            
            date_elapsed_time = time.time() - date_start_time 
            
            print(f"\n🏆 【真機結果回傳成功】")
            print(f"🎯 調倉日: {date}")
            print(f"🎯 選定投資組合: {qaoa_selected}")
            print(f"⚡ 真機計算之能量值 (含懲罰 fval): {fval:.6f}")
            print(f"💰 還原真實財務 Energy: {real_energy:.6f}")
            print(f"🆔 Job ID: {job.job_id()}")
            print(f"⏱️ 本次任務總耗時: {date_elapsed_time:.2f} 秒 (其中 IBM 真機排隊與運算佔 {job_elapsed_time:.2f} 秒)")
            
        except Exception as e:
            print(f"\n❌ 任務處理發生問題，原因: {e}")
        
        print(f"==========================================\n")

    total_elapsed_time = time.time() - total_start_time
    print(f"🏁 程式全部執行完畢！總執行時間: {total_elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()