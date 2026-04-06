import time
import numpy as np
import matplotlib.pyplot as plt

# Import the generation functions from your existing scripts
from Target_Present import monte_carlo_average, sparse_to_dense
from Target_Absent import simulate_target_absent
from Target_present_optimisation import  monte_carlo_vectorized

def run_scaling_analysis():
    # --- Fixed Global Parameters ---
    Nbar = 0.5
    Kappa = 0.05
    K_samples = 1000  # Keeping constant for the benchmark
    
    # ==========================================
    # EXPERIMENT 1: Scale M (Constant Nmax = 2)
    # ==========================================
    fixed_Nmax_exp1 = 2
    M_values = [2, 3, 4, 5,6,7,8]
    
    times_mc_exp1 = []
    times_absent_exp1 = []

    print(f"--- EXPERIMENT 1: Scaling M (Nmax = {fixed_Nmax_exp1}) ---")
    for M in M_values:
        print(f"Profiling M = {M}...")
        
        # Target Present: Monte Carlo Generation Only O(KMd^3)
        """t0 = time.time()
        rho_sparse = monte_carlo_average(M, Kappa, Nbar, fixed_Nmax_exp1, K_samples)
        t1 = time.time()
        times_mc_exp1.append(t1 - t0)
        
        t0 = time.time()
        rho_dense_pres = sparse_to_dense(rho_sparse, M, fixed_Nmax_exp1)
        t1 = time.time()
        times_dense_exp1.append(t1 - t0) """

        t0= time.time()
        blocks= monte_carlo_vectorized(M, Kappa, Nbar, fixed_Nmax_exp1, K_samples )
        t1 = time.time()
        times_mc_exp1.append(t1 - t0)


        # Target Absent: Analytical O(M * d^M)
        t0 = time.time()
        rho_dense_abs = simulate_target_absent(M, fixed_Nmax_exp1, Nbar)
        t1 = time.time()
        times_absent_exp1.append(t1 - t0)

    # ==========================================
    # EXPERIMENT 2: Scale d (Constant M = 2)
    # ==========================================
    fixed_M_exp2 = 2
    Nmax_values = [2, 3, 4, 5, 6,7,8] 
    d_values = [n + 1 for n in Nmax_values]
    
    times_mc_exp2 = []
    times_absent_exp2 = []

    print(f"\n--- EXPERIMENT 2: Scaling d (M = {fixed_M_exp2}) ---")
    for Nmax in Nmax_values:
        d = Nmax + 1
        print(f"Profiling Nmax = {Nmax} (d = {d})...")
        
        # Target Present: Monte Carlo Generation Only O(KMd^3)
        """t0 = time.time()
        rho_sparse = monte_carlo_average(fixed_M_exp2, Kappa, Nbar, Nmax, K_samples)
        t1 = time.time()
        times_mc_exp2.append(t1 - t0)
        
        t0 = time.time()
        rho_dense_pres = sparse_to_dense(rho_sparse, fixed_M_exp2, Nmax)
        t1 = time.time()
        times_dense_exp2.append(t1 - t0)"""

        t0= time.time()
        blocks= monte_carlo_vectorized(fixed_M_exp2, Kappa, Nbar, Nmax, K_samples )
        t1 = time.time()
        times_mc_exp1.append(t1 - t0)

        # Target Absent: Analytical
        t0 = time.time()
        rho_dense_abs = simulate_target_absent(fixed_M_exp2, Nmax, Nbar)
        t1 = time.time()
        times_absent_exp2.append(t1 - t0)

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Graph 1: Scaling M
    ax1.plot(M_values, times_mc_exp1, marker='o', color='green', linewidth=2, 
             label='Target Present (MC Generation O(KMd³))')
    ax1.plot(M_values, times_absent_exp1, marker='s', color='blue', 
             label='Target Absent (Analytical Construction)')
    
    ax1.set_title(f'Execution Time vs Number of Modes ($M$)\nFixed $N_{{max}}={fixed_Nmax_exp1}$ ($d={fixed_Nmax_exp1+1}$), $K={K_samples}$')
    ax1.set_xlabel('Number of Modes ($M$)')
    ax1.set_ylabel('Execution Time (seconds)')
    #ax1.set_yscale('log')
    ax1.set_xticks(M_values)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # Graph 2: Scaling d
    ax2.plot(d_values, times_mc_exp2, marker='o', color='green', linewidth=2, 
             label='Target Present (MC Generation O(KMd³))')
    ax2.plot(d_values, times_absent_exp2, marker='s', color='blue', 
             label='Target Absent (Analytical Construction)')
    
    ax2.set_title(f'Execution Time vs Mode Dimension ($d$)\nFixed $M={fixed_M_exp2}$, $K={K_samples}$')
    ax2.set_xlabel('Dimension per Mode ($d = N_{max} + 1$)')
    ax2.set_ylabel('Execution Time (seconds)')
    #ax2.set_yscale('log')
    ax2.set_xticks(d_values)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('scaling_analysis_separated.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_scaling_analysis()