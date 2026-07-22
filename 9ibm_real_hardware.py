import json
import warnings
import numpy as np
import time

# 匯入 Qiskit 最佳化與量子計算相關套件
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
    
    # 初始化一個二次規劃模型
    qp = QuadraticProgram()
    for a in ASSETS: qp.binary_var(name=a)
    
    linear = {ASSETS[i]: -0.5 * mu[i] for i in range(15)}
    quadratic = {(ASSETS[i], ASSETS[j]): 0.5 * sigma[i][j] for i in range(15) for j in range(15)}
    qp.minimize(linear=linear, quadratic=quadratic)
    
    # 🚨 修正 1：總持股數限制改為 4 檔
    # 將所有資產變數相加 (係數為 1)，必須剛好等於 4 (sense='==', rhs=4)
    qp.linear_constraint(linear={a: 1 for a in ASSETS}, sense='==', rhs=4)
    mkt_assets = {
        "US": ["NVDA", "AMD", "QCOM", "AMAT", "ASML"],
        "TW": ["2330.TW", "2454.TW", "3711.TW", "6488.TWO"],
        "JP": ["8035.T", "6857.T", "4063.T"],
        "KR": ["005930.KS", "000660.KS", "042700.KS"]
    }
    
    # 🚨 修正 2：每個市場 (美、台、日、韓) 配額限制為 1 檔
    markets = {"US": 1, "TW": 1, "JP": 1, "KR": 1}
    for mkt, req in markets.items():
        qp.linear_constraint(linear={t: 1 for t in mkt_assets[mkt]}, sense='==', rhs=req)
        
    # 處理供應鏈相依性的懲罰邏輯 (Penalty Logic)
    obj = qp.objective
    for dep, relies_on in [("NVDA", "2330.TW"), ("AMD", "2330.TW"), ("2330.TW", "ASML")]:
        obj.linear[dep] += 10
        obj.quadratic[dep, relies_on] -= 10

    return QuadraticProgramToQubo(penalty=100).convert(qp)

def calculate_original_energy(selected_tickers, mu, sigma, lmbda=0.5):
    """將真機選出的股票代回原始目標函數，計算純粹的財務 Energy (不含 QUBO 懲罰項)"""
    
    # 建立長度為 15 的全 0 陣列
    x = np.zeros(len(ASSETS))
    # 將有選中的股票位置設為 1
    for ticker in selected_tickers:
        if ticker in ASSETS:
            idx = ASSETS.index(ticker)
            x[idx] = 1

    # 計算投資組合的總風險 (x^T * Sigma * x)
    risk = float(x.T @ sigma @ x)
    # 計算投資組合的總預期報酬 (Mu * x)
    expected_return = float(np.dot(mu, x))
    
    # 回傳真實財務能量值 (值越小代表投組表現越好)
    return lmbda * risk - (1 - lmbda) * expected_return

def main():
    total_start_time = time.time()  # ✅ 新增：記錄程式總開始時間
    
    print("=== 🚀 連線至 IBM Quantum 真實量子電腦===\n")
    
    # 進行帳號驗證並儲存 Token
    print("🔑 正在驗證 IBM Token...")
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=IBM_TOKEN, set_as_default=True, overwrite=True)
    service = QiskitRuntimeService()
    
    # 自動尋找目前最空閒 (least_busy)、正在運行 (operational) 且大於等於 15 顆量子位元的「真實電腦」(非模擬器)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=15)
    
    # 建立採樣器 (Sampler)，這是在 V2 架構下負責對量子電路進行測量的工具
    sampler = Sampler(mode=backend)
    print(f"✅ 成功連線！已鎖定真實量子電腦: {backend.name} (Qubits: {backend.num_qubits})\n")

    # 嘗試載入預測報酬與共變異數的歷史數據
    try:
        with open("lstm_predicted_mu.json", 'r') as f: mu_data = json.load(f)
        with open("sigma_matrices.json", 'r') as f: sigma_data = json.load(f)
    except FileNotFoundError:
        print("❌ 錯誤：找不到數據檔案 (json)，請確認檔案路徑。")
        return
        
    # 定義實驗所要發送的特定調倉日期
    target_dates = ["2019-05-13","2019-06-19","2019-06-24","2020-03-04","2026-04-27"]

    for date in target_dates:
        date_start_time = time.time()  # ✅ 新增：記錄單次調倉日開始時間
        
        print(f"==========================================")
        print(f"📅 準備發送調倉日: {date} 之任務")
        
        # 若數據檔內沒有該日期的資料，則跳過
        if date not in mu_data: continue

        # 抓取該日期的預期報酬與風險矩陣
        mu = [mu_data[date][k] for k in ASSETS]
        sigma = np.array(sigma_data[date])
        
        # 1. 建構 QUBO 模型，並轉化為量子物理學上的 Ising Hamiltonian 算符與位移值 (offset)
        qubo_model = build_qubo_model(mu, sigma)
        observable, offset = qubo_model.to_ising() 
        
        # 2. 建立 QAOA 參數化量子電路 (Ansatz) 並加上測量閘 (Measurement)
        ansatz = QAOAAnsatz(cost_operator=observable, reps=1)
        ansatz.measure_all() 
        
        # 使用 Pass Manager，根據特定目標量子電腦的硬體結構 (拓樸)，將電路轉譯 (Transpile) 過去
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_circuit = pm.run(ansatz)

        try:
            print(f"📡 正在發送 ISA 指令至 {backend.name} (15 Qubits)...")
            
            # 設定初始角度參數 theta (給定一個初始值供單次採樣)
            initial_theta = [0.1] * ansatz.num_parameters
            
            job_start_time = time.time()  # ✅ 新增：記錄任務送出到 IBM 的時間
            
            # 發送採樣任務至 IBM Quantum 平台
            job = sampler.run([(isa_circuit, initial_theta)])
            print(f"🆔 Job ID: {job.job_id()}")
            
            # --- 監控排隊與執行狀態 ---
            print("⏳ 任務已進入 IBM 系統，正在監控進度...")
            while True:
                status = job.status()
                status_name = status.name if hasattr(status, 'name') else str(status)
                
                # 若執行完畢跳出迴圈
                if status_name == 'DONE':
                    print("\n✅ 任務執行完成！正在抓取結果數據...")
                    break
                # 若任務失敗則拋出異常
                elif status_name in ['FAILED', 'CANCELLED', 'ERROR']:
                    raise RuntimeError(f"IBM 任務異常終止，狀態: {status_name}")
                # 任務可能還在排隊 (QUEUED) 或執行中 (RUNNING)，每 30 秒確認一次
                else:
                    print(f"   [目前狀態]: {status_name} (請耐心等待...)", end='\r')
                    time.sleep(30)
            
            job_elapsed_time = time.time() - job_start_time  # ✅ 新增：計算 IBM 排隊與運算時間
            
            # --- 解析回傳結果 ---
            result = job.result() 
            pub_result = result[0]
            
            data_bin = pub_result.data
            counts = None
            
            # 遍歷 data_bin，動態找出包含 get_counts() 的欄位 (因 Qiskit 版本不同，名稱可能有異)
            for field in data_bin:
                field_data = getattr(data_bin, field)
                if hasattr(field_data, 'get_counts'):
                    counts = field_data.get_counts()
                    break
            
            if counts is None:
                raise ValueError("無法從 DataBin 中找到有效的測量計數數據")

            # 找出量子測量結果中，機率最高 (出現次數最多) 的字串組合
            best_bitstring = max(counts, key=counts.get)
            
            # Qiskit 輸出的位元順序是反向的，這裡先反轉字串，再轉為整數列表 (0 或 1)
            x_sample = [int(bit) for bit in best_bitstring[::-1]]
            
            # 將這個位元組合帶入 QUBO 模型，計算包含懲罰項的目標值 (fval)
            fval = qubo_model.objective.evaluate(x_sample)
            
            # 將位元陣列轉換回對應的資產代碼清單
            qaoa_selected = [ASSETS[i] for i, val in enumerate(x_sample) if val == 1]
            
            # 🚨 修正 3：計算該投組「純粹的財務 Energy」，將剛剛算出的名單帶入函式
            real_energy = calculate_original_energy(qaoa_selected, mu, sigma)
            
            date_elapsed_time = time.time() - date_start_time  # ✅ 新增：計算該日期總耗時
            
            print(f"\n🏆 【真機結果回傳成功】")
            print(f"🎯 調倉日: {date}")
            print(f"🎯 選定投資組合: {qaoa_selected}")
            print(f"⚡ 真機計算之能量值 (含懲罰 fval): {fval:.6f}")
            print(f"💰 還原真實財務 Energy: {real_energy:.6f}")
            print(f"🆔 Job ID: {job.job_id()}")
            # ✅ 新增：印出運算時間資訊
            print(f"⏱️ 本次任務總耗時: {date_elapsed_time:.2f} 秒 (其中 IBM 真機排隊與運算佔 {job_elapsed_time:.2f} 秒)")
            
        except Exception as e:
            print(f"\n❌ 任務處理發生問題，原因: {e}")
        
        print(f"==========================================\n")

    total_elapsed_time = time.time() - total_start_time  # ✅ 新增：計算程式全部執行完畢耗時
    print(f"🏁 程式全部執行完畢！總執行時間: {total_elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()