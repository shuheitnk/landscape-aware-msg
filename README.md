
# Designing Landscape-Aware Benchmarks with Explicit Local Optima for Single- and Multi-Objective Optimization

---

## 1. Prerequisites

- **Operating System:** Windows 10/11 64-bit
- **Python:** 3.11.9
- **GPU:** NVIDIA GPU with CUDA 12.1+ 
- **VS Code**
> All commands below assume execution in the **PowerShell terminal**.

---

## 2. Install VS Code

Download and install from: https://code.visualstudio.com/download  

---

## 3. Install Python 3.11.9

Download installer: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

Verify installation:

```powershell
python --version
# Expected: Python 3.11.9
```

---

## 4. Install NVIDIA Driver (if using GPU)

Download: https://www.nvidia.com/en-us/drivers/  

- Choose **Game Studio Driver**
- Verify installation:

```powershell
nvidia-smi
```

---

## 5. Install CUDA Toolkit 12.1.1

Download: https://developer.nvidia.com/cuda-12-1-1-download-archive  

Verify installation:

```powershell
nvcc --version
```

---

## 6. Clone the Repository

```powershell
git clone https://github.com/shuheitnk/landscape-aware-msg.git
cd landscape-aware-msg
```

---

## 7. Create and Activate a Python Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python --version  # should show 3.11.9
```

If you encounter a **PSSecurityException (Execution Policy Restriction)**, enable script execution for your user only:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

## 8. Install Dependencies

Install PyTorch with CUDA support:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Install the remaining requirements:

```powershell
pip install -r requirements.txt
```

---

## 10. Trace our experiments

### 10.1 Trace experiment for RQ1
```powershell
python .\Experiment_RQ1\bbob_fitting.py --D 2 --out \res_rq1\results_2d.csv --num_gaussians 100 --pop_size 200 --generations 200
python .\Experiment_RQ1\bbob_fitting.py --D 5 --out \res_rq1\results_5d.csv --num_gaussians 250 --pop_size 200 --generations 200
python .\Experiment_RQ1\bbob_fitting.py --D 10 --out \res_rq1\results_10d.csv --num_gaussians 500 --pop_size 200 --generations 200
```

### 10.2 Trace experiment for RQ2
```powershell
python .\Experiment_RQ2\search_feature_range.py --D 2 --out_dir .\res_rq2 --generations 200 --pop_size 200 --num_runs 11 --num_gaussians 100
python .\Experiment_RQ2\search_feature_range.py --D 5 --out_dir .\res_rq2 --generations 200 --pop_size 200 --num_runs 11 --num_gaussians 250
python .\Experiment_RQ2\search_feature_range.py --D 10 --out_dir .\res_rq2 --generations 200 --pop_size 200 --num_runs 11 --num_gaussians 500 
python .\Experiment_RQ2\culc_bbob_ela_feature_vec.py --D 2 --sampling_factor 500 --out_dir .\res_rq2\bbob_ela --num_runs 11 --max_functions 24 --max_instances 10
python .\Experiment_RQ2\culc_bbob_ela_feature_vec.py --D 5 --sampling_factor 500 --out_dir .\res_rq2\bbob_ela --num_runs 11 --max_functions 24 --max_instances 10
python .\Experiment_RQ2\culc_bbob_ela_feature_vec.py --D 10 --sampling_factor 500 --out_dir .\res_rq2\bbob_ela --num_runs 11 --max_functions 24 --max_instances 10
python .\Experiment_RQ2\culc_msg_ela_feature_vec.py --D 2 --num_gaussians 100  --base_path .\res_rq2 --out_dir .\res_rq2\msg_ela --pop_size 200 --generations 200 --sampling_factor 500 --num_runs 11
python .\Experiment_RQ2\culc_msg_ela_feature_vec.py --D 5 --num_gaussians 250  --base_path .\res_rq2 --out_dir .\res_rq2\msg_ela --pop_size 200 --generations 200 --sampling_factor 500 --num_runs 11
python .\Experiment_RQ2\culc_msg_ela_feature_vec.py --D 10 --num_gaussians 500  --base_path .\res_rq2 --out_dir .\res_rq2\msg_ela --pop_size 200 --generations 200 --sampling_factor 500 --num_runs 11
run .\Experiment_RQ2\extract_stable_features.ipynb
run .\Experiment_RQ2\pca_feature_scpace.ipynb
```

### 10.3 Trace experiment for RQ3
```powershell
python .\Experiment_RQ3\culc_multi_msg_s_ela_feature_vec.py --function_id 222 --result_path ..\res_rq2\msg_ela\results_max_max_max_2d.pt --out_dir ..\res_rq3\out_s_ela --D 2 --num_gaussians 100 --sampling_factor 100 --num_runs 11
```
Run multi-objective MSG ELA feature computation for all function--function_id 111, 112, 121, 122, 211, 212, 221, 222
```powershell
run .\Experiment_RQ3\culc_correlations_ela_s_ela.ipynb
```