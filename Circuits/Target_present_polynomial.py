import numpy as np
import itertools
from scipy.linalg import expm
from Target_Present import bose_einstein_pmf, thermal_distribution, sample_environment,beam_splitter_fock, compute_v_states, local_bs_data



def generate_block_basis(M, d):
    """Pre-computes the valid basis states for each Nc block."""
    basis_by_Nc = {}
    # itertools.product ensures exact lexicographical ordering to match old matrix
    all_n_dists = list(itertools.product(range(d), repeat=M))
    
    for k in range(M):
        for n_tuple in all_n_dists:
            Nc = sum(n_tuple)
            if Nc not in basis_by_Nc:
                basis_by_Nc[Nc] = []
            basis_by_Nc[Nc].append((k, n_tuple))
            
    return basis_by_Nc

def compute_mode_sigmas(v0_dict, v1_dict, d):
    """Converts sparse v0/v1 dicts to dense mode sigmas (Tracing out Env)."""
    v0_dense = np.zeros((d, d), dtype=complex)
    v1_dense = np.zeros((d, d), dtype=complex)
    
    for (s, e), amp in v0_dict.items():
        v0_dense[s, e] = amp
    for (s, e), amp in v1_dict.items():
        v1_dense[s, e] = amp
        
    sigmas = {}
    # Tr_E[ |v(q')><v(q)| ] = v_qprime @ v_q.conj().T
    sigmas[(0,0)] = v0_dense @ v0_dense.conj().T
    sigmas[(1,1)] = v1_dense @ v1_dense.conj().T
    sigmas[(0,1)] = v0_dense @ v1_dense.conj().T
    sigmas[(1,0)] = v1_dense @ v0_dense.conj().T
    
    return sigmas

def monte_carlo_direct_to_blocks(M, Kappa, Nbar, Nmax, K_samples):
    d = Nmax + 1
    
    # Initialize Blocks
    basis_by_Nc = generate_block_basis(M, d)
    blocks = {}
    for Nc, basis_list in basis_by_Nc.items():
        dim = len(basis_list)
        blocks[Nc] = np.zeros((dim, dim), dtype=complex)
        
    print(f"Starting Monte Carlo Direct-to-Blocks ({K_samples} samples)...")
    
    # Monte Carlo Loop
    for sample in range(K_samples):
        # 1. Physics Engine: Get local outputs
        n_vec = sample_environment(M, Nbar, Nmax)
        v0_list, v1_list = local_bs_data(n_vec, Kappa, Nmax)
        
        # 2. Compute Mode Sigmas
        sigmas_all_modes = []
        for j in range(M):
            sigmas_j = compute_mode_sigmas(v0_list[j], v1_list[j], d)
            sigmas_all_modes.append(sigmas_j)
            
        # 3. Direct Block Population
        for Nc, basis_list in basis_by_Nc.items():
            dim = len(basis_list)
            for row_idx in range(dim):
                k_prime, n_prime = basis_list[row_idx]
                for col_idx in range(dim):
                    k, n = basis_list[col_idx]
                    
                    val = 1.0
                    for j in range(M):
                        q_prime = 1 if j == k_prime else 0
                        q = 1 if j == k else 0
                        # Extract and multiply the local matrix elements
                        val *= sigmas_all_modes[j][(q_prime, q)][n_prime[j], n[j]]
                        
                    blocks[Nc][row_idx, col_idx] += val / M
                    
    # Average the blocks
    for Nc in blocks:
        blocks[Nc] /= K_samples
        
    return blocks

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- Parameters ---
    M = 5
    Nmax = 2
    Nbar = 0.5
    Kappa = 0.05 
    K_samples = 1000 
    
    print(f"Simulating Target Present (M={M}, kappa={Kappa})...")
    
    # Run the new optimized loop
    blocks = monte_carlo_direct_to_blocks(M, Kappa, Nbar, Nmax, K_samples)
    
    print(f"\nSimulation Complete. Generated Block Matrices: {blocks}")
    
    total_trace = 0.0
    for Nc, matrix in blocks.items():
        block_trace = np.real(np.trace(matrix))
        total_trace += block_trace
        print(f"\nBlock Nc = {Nc} | Shape: {matrix.shape} | Trace contribution: {block_trace:.4f}")
        print(np.round(matrix, 5)) # Print the block
        
    print(f"\nTotal Trace across all blocks: {total_trace:.4f} (Should be 1.0)")