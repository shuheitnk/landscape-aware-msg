import argparse
from collections import defaultdict
from typing import List, Tuple
import logging
import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path
import sys
import re

# -------------------------------
# Add parent folder to path
# -------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from Experiment_RQ2 import create_msg_samples
from x_msg.construct_msg_landscape import MSGLandscape
from x_msg.sampling import sobol_sampling
from pflacco import classical_ela_features, misc_features

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("S-ELA")

# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed: int, device="cuda") -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------
# Compute ELA features
# -------------------------------
def compute_ela_features(X: torch.Tensor, y: torch.Tensor) -> dict:
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    y_min, y_max = y_np.min(), y_np.max()
    y_norm = (y_np - y_min) / (y_max - y_min) if y_max - y_min > 0 else y_np*0.0
    feats = {
        **classical_ela_features.calculate_ela_distribution(X, y_norm),
        **classical_ela_features.calculate_ela_level(X, y_norm),
        **classical_ela_features.calculate_ela_meta(X, y_norm),
        **classical_ela_features.calculate_pca(X, y_norm),
        **classical_ela_features.calculate_nbc(X, y_norm),
        **classical_ela_features.calculate_dispersion(X, y_norm),
        **classical_ela_features.calculate_information_content(X, y_norm),
        **misc_features.calculate_fitness_distance_correlation(X, y_norm),
    }
    return {k:v for k,v in feats.items() if not k.endswith("costs_runtime")}

# -------------------------------
# Simplex lattice for 2D
# -------------------------------
def simplex_lattice_2d(num_points: int=5) -> List[Tuple[float,float]]:
    H = num_points - 1
    return [(i/H, (H-i)/H) for i in range(H+1)]

# -------------------------------
# Non-dominated rank normalized
# -------------------------------
def normalized_pareto_rank(Y: np.ndarray) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Y_t = torch.tensor(Y, device=device)
    Y_i = Y_t.unsqueeze(1)
    Y_j = Y_t.unsqueeze(0)
    le = (Y_j <= Y_i).all(dim=2)
    lt = (Y_j < Y_i).any(dim=2)
    dominated = le & lt
    dominated.fill_diagonal_(False)
    count = dominated.sum(dim=1).float()
    maxv = count.max()
    if maxv>0:
        count = count/maxv
    return count.cpu().numpy()

# -------------------------------
# Process single repetition
# -------------------------------
def process_single_repeat(function_id: int, rep:int, rep_seed:int, result, means:torch.Tensor, W:List[Tuple[float,float]],
                          D:int, sampling_factor:int, device:str) -> pd.DataFrame:
    logger.info(f"Processing repetition {rep} with seed {rep_seed}")
    set_seed(rep_seed, device=device)
    X, Y1, Y2 = create_msg_samples.create_bi_msg_samples(
        result=result, means=means, MSGLandscape=MSGLandscape,
        D=D, sampling_factor=sampling_factor, device=device, seed=rep_seed
    )
    print(Y1.columns)
    print(Y2.columns)

    df_rep = pd.DataFrame()
    for key in Y1.columns:
        per_weight_features = []
        for w1, w2 in W:
            y = w1*Y1[key] + w2*Y2[key]
            feats = compute_ela_features(X, y)
            per_weight_features.append(feats)
        df_w = pd.DataFrame(per_weight_features)
        stat_vec = pd.concat([
            df_w.min().add_suffix("_min"),
            df_w.median().add_suffix("_median"),
            df_w.max().add_suffix("_max"),
            df_w.std().add_suffix("_sd")
        ])
        Y_multi = np.stack([Y1[key].to_numpy(), Y2[key].to_numpy()], axis=1)
        feats_domi = pd.Series(compute_ela_features(X, normalized_pareto_rank(Y_multi))).add_suffix("_domi")
        full_vec = pd.concat([stat_vec, feats_domi])
        df_row = pd.DataFrame([full_vec])
        df_row.insert(0, "seed", rep_seed)
        df_row.insert(0, "instance_id", int(re.search(r"\d+", key).group()))
        df_row.insert(0, "function_id", int(function_id))
        df_rep = pd.concat([df_rep, df_row], axis=0, ignore_index=True)
    return df_rep

# -------------------------------
# Argument parser
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="S-ELA: Compute convex combination ELA features")
    parser.add_argument("--function_id", type=int, default=1)
    parser.add_argument("--result_path", type=str, default="../res_rq2/msg_ela/results_max_max_max_2d.pt",
                        help="Relative path to MSG .pt file from script location")
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--num_gaussians", type=int, default=100)
    parser.add_argument("--sampling_factor", type=int, default=500)
    parser.add_argument("--num_runs", type=int, default=11)
    parser.add_argument("--num_weight", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="../res_rq2/out_s_ela",
                        help="Relative output folder from script location")
    return parser.parse_args()

# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, device=device)
    logger.info(f"Using device: {device}")

    result_path = (SCRIPT_DIR / args.result_path).resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"{result_path} not found")
    result = torch.load(result_path, map_location=device)
    logger.info(f"Loaded MSG result from {result_path}")

    means = sobol_sampling(dim=args.D, num_samples=args.num_gaussians, device=device, seed=args.seed)
    W = simplex_lattice_2d(args.num_weight)
    out_dir = (SCRIPT_DIR / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.DataFrame()
    features_per_ykey = defaultdict(list)
    for rep in range(1, args.num_runs+1):
        df_rep = process_single_repeat(
            function_id=args.function_id,
            rep=rep,
            rep_seed=args.seed+rep,
            result=result,
            means=means,
            W=W,
            D=args.D,
            sampling_factor=args.sampling_factor,
            device=device
        )
        df_all = pd.concat([df_all, df_rep], axis=0)
        for _, row in df_rep.iterrows():
            features_per_ykey[row["instance_id"]].append(row)

    out_all = out_dir / f"s_ela_function_{args.function_id}_all.csv"
    df_all.to_csv(out_all, index=False)
    logger.info(f"[SAVED] All repetitions → {out_all}")

    df_median_list = []
    for k, v in features_per_ykey.items():
        df_k = pd.DataFrame(v)
        numeric_cols = df_k.select_dtypes(include=[np.number]).columns
        med = df_k[numeric_cols].median(axis=0)
        med["function_id"] = int(args.function_id)
        med["instance_id"] = int(k)
        med["seed"] = int(med["seed"])
        df_median_list.append(med)

    df_median = pd.DataFrame(df_median_list).sort_values("instance_id").reset_index(drop=True)
    df_median[["function_id", "instance_id", "seed"]] = df_median[["function_id", "instance_id", "seed"]].astype(int)
    out_median = out_dir / f"s_ela_msg_{args.function_id}_{args.D+1}d_median.csv"
    df_median.to_csv(out_median, index=False)
    logger.info(f"[SAVED] Median features → {out_median}")

if __name__=="__main__":
    main()
