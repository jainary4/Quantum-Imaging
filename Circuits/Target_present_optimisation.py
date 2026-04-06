import numpy as np
import itertools
from scipy.linalg import expm
from Target_Present import bose_einstein_pmf, thermal_distribution, sample_environment,beam_splitter_fock, compute_v_states, local_bs_data
import time


def bose_einstein_pmf(n, nbar):
    return (nbar**n) / ((nbar + 1)**(n + 1))

def thermal_distribution(nbar, Nmax):
    probs = np.array([bose_einstein_pmf(n, nbar) for n in range(Nmax + 1)])
    probs /= probs.sum()
    return probs

def sample_environment(M, nbar, Nmax):
    probs = thermal_distribution(nbar, Nmax)
    return np.random.choice(np.arange(Nmax + 1), size=M, p=probs)

def beam_splitter_fock(nS, nE, eta, Nmax):
    amps = {}
    N = nS + nE 
    dim = N + 1
    
    G = np.zeros((dim, dim))
    for k in range(N):
        val = np.sqrt(k + 1) * np.sqrt(N - k)
        G[k + 1, k] = val
        G[k, k + 1] = -val

    theta = np.arccos(np.sqrt(eta))
    U = expm(theta * G)
    
    input_idx = nS
    for k_out in range(dim):
        amp = U[k_out, input_idx]
        nS_out = k_out
        nE_out = N - k_out
        
        if nS_out <= Nmax and nE_out <= Nmax:
            if abs(amp) > 1e-12:
                amps[(nS_out, nE_out)] = amp
    return amps

def compute_v_states(n_env, eta, Nmax):
    v0 = beam_splitter_fock(nS=0, nE=n_env, eta=eta, Nmax=Nmax)
    v1 = beam_splitter_fock(nS=1, nE=n_env, eta=eta, Nmax=Nmax)
    return v0, v1

def local_bs_data(n_vec, eta, Nmax):
    v0_list = []
    v1_list = []
    for n_j in n_vec:
        v0, v1 = compute_v_states(n_j, eta, Nmax)
        v0_list.append(v0)
        v1_list.append(v1)
    return v0_list, v1_list


def generate_block_basis_arrays(M, d):
    """
    Pre-computes basis states, but formats them as flat NumPy arrays 
    so they can be fed directly into C-level advanced indexing.
    """
    basis_by_Nc = {}
    all_n_dists = np.array(list(itertools.product(range(d), repeat=M)))
    sum_n = np.sum(all_n_dists, axis=1)
    
    for k in range(M):
        for i in range(len(all_n_dists)):
            n_tuple = all_n_dists[i]
            Nc = sum_n[i]
            if Nc not in basis_by_Nc:
                basis_by_Nc[Nc] = {'k': [], 'n': []}
            basis_by_Nc[Nc]['k'].append(k)
            basis_by_Nc[Nc]['n'].append(n_tuple)
            
    # Convert lists to NumPy arrays
    for Nc in basis_by_Nc:
        basis_by_Nc[Nc]['k'] = np.array(basis_by_Nc[Nc]['k'])
        basis_by_Nc[Nc]['n'] = np.array(basis_by_Nc[Nc]['n'])
        
    return basis_by_Nc

def compute_mode_sigmas(v0_dict, v1_dict, d):
    v0_dense = np.zeros((d, d), dtype=complex)
    v1_dense = np.zeros((d, d), dtype=complex)
    
    for (s, e), amp in v0_dict.items():
        v0_dense[s, e] = amp
    for (s, e), amp in v1_dict.items():
        v1_dense[s, e] = amp
        
    sigmas = {}
    sigmas[(0,0)] = v0_dense @ v0_dense.conj().T
    sigmas[(1,1)] = v1_dense @ v1_dense.conj().T
    sigmas[(0,1)] = v0_dense @ v1_dense.conj().T
    sigmas[(1,0)] = v1_dense @ v0_dense.conj().T
    return sigmas

def monte_carlo_vectorized(M, Kappa, Nbar, Nmax, K_samples):
    d = Nmax + 1
    
    # 1. Initialize Blocks and Pre-calculate Broadcast Coordinates
    basis_by_Nc = generate_block_basis_arrays(M, d)
    blocks = {Nc: np.zeros((len(data['k']), len(data['k'])), dtype=complex) 
              for Nc, data in basis_by_Nc.items()}
              
    broadcast_indices = {}
    j_idx = np.arange(M).reshape(1, 1, M) # Spatial mode axis
    
    for Nc, data in basis_by_Nc.items():
        dim = len(data['k'])
        if dim == 0: continue
        
        # Format the Bra (Row) coordinates to broadcast downwards: shape (dim, 1, M)
        k_prime = data['k'].reshape(dim, 1,1)
        n_prime = data['n'].reshape(dim, 1, M)
        Q_prime = (np.arange(M) == k_prime).astype(int)
        
        # Format the Ket (Col) coordinates to broadcast rightwards: shape (1, dim, M)
        k_ket = data['k'].reshape(1, dim,1)
        n_ket = data['n'].reshape(1, dim, M)
        Q_ket = (np.arange(M) == k_ket).astype(int)
        
        broadcast_indices[Nc] = (Q_prime, Q_ket, n_prime, n_ket)
        
    print(f"Starting Vectorized Monte Carlo ({K_samples} samples)...")
    
    # 2. Monte Carlo Loop
    for sample in range(K_samples):
        n_vec = sample_environment(M, Nbar, Nmax)
        v0_list, v1_list = local_bs_data(n_vec, Kappa, Nmax)
        
        # Pack all local sigmas into a single 5D C-array: Shape (M, 2, 2, d, d)
        S = np.zeros((M, 2, 2, d, d), dtype=complex)
        for j in range(M):
            sigmas = compute_mode_sigmas(v0_list[j], v1_list[j], d)
            S[j, 0, 0] = sigmas[(0,0)]
            S[j, 1, 1] = sigmas[(1,1)]
            S[j, 0, 1] = sigmas[(0,1)]
            S[j, 1, 0] = sigmas[(1,0)]
            
        # 3. Vectorized Block Population
        for Nc, (Q_prime, Q_ket, n_prime, n_ket) in broadcast_indices.items():
            
            # The Magic Line: Advanced indexing extracts every scalar for the entire block simultaneously
            extracted_vals = S[j_idx, Q_prime, Q_ket, n_prime, n_ket]
            
            # Multiply the scalars across the spatial mode axis (axis=-1)
            block_vals = np.prod(extracted_vals, axis=-1)
            
            # Add the fully calculated grid to the block
            blocks[Nc] += block_vals / M
            
    # 4. Average and Correct Truncation Norm
    total_trace = 0.0
    for Nc in blocks:
        blocks[Nc] /= K_samples
        total_trace += np.real(np.trace(blocks[Nc]))
        
    for Nc in blocks:
        blocks[Nc] /= total_trace # Normalization fix for Fock truncation
        
    return blocks

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    M = 2
    Nmax = 2
    Nbar = 0.5
    Kappa = 0.05 
    K_samples = 1000 
    
    print(f"Simulating Target Present Vectorized (M={M}, kappa={Kappa})...")

    t0= time.time()
    
    blocks = monte_carlo_vectorized(M, Kappa, Nbar, Nmax, K_samples)

    t1= time.time()

    print(f"\nSimulation Complete. Generated Block Matrices: {blocks}")

    print(f"The time taken for the simualtion is : {t1-t0}")
    
    total_trace = 0.0
    for Nc, matrix in blocks.items():
        block_trace = np.real(np.trace(matrix))
        total_trace += block_trace
        print(f"\nBlock Nc = {Nc} | Shape: {matrix.shape} | Trace contribution: {block_trace:.4f}")
        print(np.round(matrix, 5)) # Uncomment to view matrices
        
    print(f"\nTotal Trace across all blocks: {total_trace:.4f} (Should be 1.0)")