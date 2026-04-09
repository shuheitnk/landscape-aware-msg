# =====================================================
# Imports
# =====================================================
import time
import random
import argparse

import numpy as np
import torch
import cocoex
from pyDOE import lhs
from scipy.stats import spearmanr, test

import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from x_msg.evolution_strategy import EvolutionStrategy
from x_msg.sampling import sobol_sampling


# =====================================================
# Reproducibility Utilities
# =====================================================
def set_seed(seed: int, device="cuda") -> None:
    """
    Sets random seeds for reproducibility.

    device: str or torch.device
    """
    import torch, random, numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev_str = str(device) if isinstance(device, torch.device) else device

    if dev_str.startswith("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# 1. MSG Landscape Evaluator (theta-parallel)
# =====================================================
class MSGLandscapeEvaluator(torch.nn.Module):
    """
    Evaluates a family of MSG landscapes in parallel for fixed evaluation points X.

    Parallelization is over theta (landscape parameters), not over X.
    """

    def __init__(self, means: torch.Tensor, chunk_size: int = 32):
        super().__init__()
        self.register_buffer("means", means)
        self.register_buffer("sq_dist_base", None)

        self.num_gaussians, self.dim = means.shape
        self.chunk_size = chunk_size

    @torch.no_grad()
    def set_X(self, X: torch.Tensor) -> None:
        """
        Precompute squared distances between evaluation points X and Gaussian centers.
        """
        X_norm2 = (X ** 2).sum(dim=1, keepdim=True)           # (N, 1)
        M_norm2 = (self.means ** 2).sum(dim=1).unsqueeze(0)  # (1, M)
        sq_dist = X_norm2 + M_norm2 - 2.0 * (X @ self.means.T)
        self.sq_dist_base = sq_dist.unsqueeze(0)             # (1, N, M)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Evaluate MSG landscapes for a batch of parameter vectors.
        """
        assert self.sq_dist_base is not None, "set_X must be called before forward()."

        outputs = []
        B = thetas.shape[0]
        M = self.num_gaussians

        for i in range(0, B, self.chunk_size):
            theta = thetas[i:i + self.chunk_size]

            alphas = theta[:, :M].unsqueeze(1)  # (b, 1, M)
            sigmas = theta[:, M:].unsqueeze(1)  # (b, 1, M)

            neg_half_inv_var = -0.5 / (sigmas ** 2)
            vals = alphas * torch.exp(self.sq_dist_base * neg_half_inv_var)

            outputs.append(-vals.max(dim=2).values)  # (b, N)

        return torch.cat(outputs, dim=0)


# =====================================================
# 2. Rank-based Fitness Function
# =====================================================
def make_rank_mse_fitness(
    msg: MSGLandscapeEvaluator,
    y_target_rank: np.ndarray,
    device: str,
):
    """
    Rank-based MSE between MSG predictions and target BBOB ranks.
    """
    y_true = torch.tensor(
        y_target_rank, dtype=torch.float32, device=device
    ).view(1, -1)

    def fitness_fn(thetas: torch.Tensor) -> torch.Tensor:
        y_pred = msg(thetas)
        ranks = torch.argsort(torch.argsort(y_pred, dim=1), dim=1)
        y_rank = ranks.float() / (y_pred.size(1) - 1)
        return (y_rank - y_true).pow(2).mean(dim=1)

    return fitness_fn

def lhs_with_seed(seed, D, samples):
    np.random.seed(seed)
    X = lhs(D, samples=samples)
    return X

# =====================================================
# 3. Entry Point
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Rank-based MSG fitting to BBOB functions using ES"
    )
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--num_gaussians", type=int, default=None)
    parser.add_argument("--pop_size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--search_method", type=str, default="es")
    parser.add_argument("--test", type=bool, default=False, help="Evaluate on different points from points that are used for fitting")

    args = parser.parse_args()

    D = args.D
    RESULT_CSV = args.out
    MEDIAN_CSV = RESULT_CSV.replace(".csv", "_median.csv")
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MEDIAN_CSV), exist_ok=True)

    NUM_GAUSSIANS = args.num_gaussians
    NUM_SAMPLES = 500 * D
    POP_SIZE = args.pop_size
    GENERATIONS = args.generations

    MUT_STD = 0.1 * (D ** 0.5) / (NUM_GAUSSIANS ** 0.5)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SEED_BASE = 42

    # -------------------------------------------------
    # CSV headers
    # -------------------------------------------------
    with open(RESULT_CSV, "w") as f:
        f.write(
            "function,seed,dim,num_gaussians,num_samples,"
            "population,generations,spearman,time_sec\n"
        )

    with open(MEDIAN_CSV, "w") as f:
        f.write(
            "function,dim,num_gaussians,num_samples,"
            "population,generations,median_spearman\n"
        )

    # -------------------------------------------------
    # BBOB functions F1--F24
    # -------------------------------------------------
    for fun_id in range(1, 25):
        problem = cocoex.BareProblem(
            "bbob", function=fun_id, dimension=D, instance=1
        )

        rho_list = []

        for run in range(11):
            seed = SEED_BASE + run
            set_seed(seed, DEVICE)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            # ---------------------------------------------
            # Sample evaluation points
            # ---------------------------------------------
            X_train = lhs_with_seed(seed+1, D, samples=NUM_SAMPLES)  # For training
            X_test = lhs_with_seed(seed+2, D, samples=NUM_SAMPLES)  # For testing
            X_bbob_train = X_train * 10.0 - 5.0
            X_bbob_test = X_test * 10.0 - 5.0
            y_train = problem(X_bbob_train).squeeze()
            y_test = problem(X_bbob_test).squeeze()

            ranks_train = y_train.argsort().argsort().astype(np.float32)
            y_rank_train = ranks_train / (len(y_train) - 1)

            ranks_test = y_test.argsort().argsort().astype(np.float32)
            y_rank_test = ranks_test / (len(y_test) - 1)

            # ---------------------------------------------
            # Construct MSG evaluator
            # ---------------------------------------------
            means = torch.tensor(
                sobol_sampling(D, NUM_GAUSSIANS, device=DEVICE, seed=seed),
                dtype=torch.float32,
                device=DEVICE,
            )

            msg = MSGLandscapeEvaluator(means)
            msg.set_X(torch.tensor(X_train, dtype=torch.float32, device=DEVICE))

            fitness_fn = make_rank_mse_fitness(msg, y_rank_train, DEVICE)

            # ---------------------------------------------
            # Initialize theta
            # ---------------------------------------------
            theta_init = torch.cat(
                [
                    torch.ones(NUM_GAUSSIANS, device=DEVICE),
                    torch.full(
                        (NUM_GAUSSIANS,),
                        0.05 * (D ** 0.5),
                        device=DEVICE,
                    ),
                ]
            )

            # ---------------------------------------------
            # Evolution Strategy
            # ---------------------------------------------
            es = EvolutionStrategy(
                2 * NUM_GAUSSIANS,
                POP_SIZE,
                MUT_STD,
                fitness_fn,
                D,
                DEVICE,
                seed=seed,
            )

            if args.search_method == "es":
                res = es.run_vanilla_es(theta_init, GENERATIONS)
            else:
                # as a random generator
                res = es.run_random_search(theta_init, GENERATIONS) 
                
            best_theta = res["theta_best"]

            # ---------------------------------------------
            # Final evaluation
            # ---------------------------------------------
            if test == True:
                with torch.no_grad():
                    msg.set_X(torch.tensor(X_test, dtype=torch.float32, device=DEVICE))
                    y_pred = msg(best_theta.unsqueeze(0))
                    rp = torch.argsort(torch.argsort(y_pred, dim=1), dim=1)
                    y_pred_rank = (
                        rp.float() / (y_pred.size(1) - 1)
                    ).cpu().numpy().squeeze()

                rho, _ = spearmanr(y_rank_test, y_pred_rank)
                rho_list.append(rho)

            else:
                with torch.no_grad():
                    y_pred = msg(best_theta.unsqueeze(0))
                    rp = torch.argsort(torch.argsort(y_pred, dim=1), dim=1)
                    y_pred_rank = (
                        rp.float() / (y_pred.size(1) - 1)
                    ).cpu().numpy().squeeze()

                rho, _ = spearmanr(y_rank_train, y_pred_rank)
                rho_list.append(rho)


            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            print(
                f"[F{fun_id:02d} | Seed {seed}] "
                f"ρ={rho:.4f} | {elapsed:.1f}s"
            )

            with open(RESULT_CSV, "a") as f:
                f.write(
                    f"{fun_id},{seed},{D},{NUM_GAUSSIANS},{NUM_SAMPLES},"
                    f"{POP_SIZE},{GENERATIONS},{rho:.6f},{elapsed:.3f}\n"
                )

        # ---------------------------------------------
        # Function-wise median
        # ---------------------------------------------
        rho_median = float(np.median(rho_list))

        print(f"[F{fun_id:02d} | MEDIAN] ρ={rho_median:.4f}")

        with open(MEDIAN_CSV, "a") as f:
            f.write(
                f"{fun_id},{D},{NUM_GAUSSIANS},{NUM_SAMPLES},"
                f"{POP_SIZE},{GENERATIONS},{rho_median:.6f}\n"
            )


if __name__ == "__main__":
    main()

