import numpy as np
from math import comb, sqrt
from scipy.linalg import expm
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

def partial_trace(rho, dims, keep):
    """Partial trace over subsystems not in 'keep'."""
    dims = list(dims)
    rho_shape = rho.shape[0]
    if rho_shape != int(np.prod(dims)):
        raise ValueError(f"Rho dimension {rho_shape} != product of dims {dims}")
    
    reshaped = rho.reshape(dims + dims)
    axis_to_trace = [i for i in range(len(dims)) if i not in keep]
    
    for ax in sorted(axis_to_trace, reverse=True):
        reshaped = np.trace(reshaped, axis1=ax, axis2=ax + len(dims))
        
    final_dim = int(np.prod([dims[i] for i in keep]))
    return reshaped.reshape(final_dim, final_dim)



def beam_splitter_fock(nS, nE, eta, Nmax):
    """
    Computes output amplitudes for a BS with transmissivity eta (kappa)
    using the Matrix Exponential method to guarantee unitarity.
    
    Input: |nS>_signal |nE>_environment
    """
    amps = {}
    N = nS + nE # Total photon number is conserved
    
    # If the total energy N exceeds our simulation truncation Nmax,
    # we physically lose information (truncation error).
    # But usually we want to capture whatever fits in Nmax.
    
    # We build the generator for the subspace of N photons.
    # Basis states are ordered by Signal photon count k:
    # Index 0: |0>_S |N>_E
    # Index 1: |1>_S |N-1>_E
    # ...
    # Index N: |N>_S |0>_E
    dim = N + 1
    
    # Generator matrix G = (a^dag_S a_E - a_S a^dag_E)
    G = np.zeros((dim, dim))
    for k in range(N):
        # Transition between |k>_S and |k+1>_S
        # sqrt(k+1) comes from a^dag_S, sqrt(N-k) comes from a_E
        val = np.sqrt(k + 1) * np.sqrt(N - k)
        
        # G[row, col]: Maps col -> row
        G[k + 1, k] = val
        G[k, k + 1] = -val

    # The mixing angle theta. cos(theta) = sqrt(eta)
    theta = np.arccos(np.sqrt(eta))
    
    # U = exp(theta * G)
    U = expm(theta * G)
    
    # The input state index corresponds to nS (Signal photons)
    input_idx = nS
    
    # Extract the column corresponding to our input
    # and map it to dictionary outputs
    for k_out in range(dim):
        amp = U[k_out, input_idx]
        
        # Map index k_out back to physical photon numbers
        nS_out = k_out
        nE_out = N - k_out
        
        # Apply truncation (optional, but consistent with Nmax)
        if nS_out <= Nmax and nE_out <= Nmax:
            if abs(amp) > 1e-12:
                amps[(nS_out, nE_out)] = amp

    return amps

def compute_v_states(n_env, eta, Nmax):
    """
    Calculates the evolution of the two superposition branches 
    for a specific environment photon count n_env.
    v(0): Input was |0>_S |n_env>_E
    v(1): Input was |1>_S |n_env>_E
    """
    # Branch 0: Signal had 0 photons
    v0 = beam_splitter_fock(nS=0, nE=n_env, eta=eta, Nmax=Nmax)
    
    # Branch 1: Signal had 1 photon
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


def tensor_product_states(state1, state2):
    out = {}
    for k1, a1 in state1.items():
        for k2, a2 in state2.items():
            # k1 is tuple, k2 is tuple -> k1+k2 concatenates them
            out[k1 + k2] = out.get(k1 + k2, 0) + a1 * a2
    return out

def build_phi_k(k, v0_list, v1_list):
    """
    Builds the state of the S+E system given that the Idler is in mode k.
    This corresponds to tensor product of v1 on mode k and v0 on all others.
    """
    phi = {(): 1.0}
    M = len(v0_list)
    for j in range(M):
        local = v1_list[j] if j == k else v0_list[j]
        # Convert local dict {(nS, nE): amp} to format suited for tensor prod
        local_state = {(nS, nE): amp for (nS, nE), amp in local.items()}
        phi = tensor_product_states(phi, local_state) # makes the implementation inefficient
    return phi

def build_output_state(v0_list, v1_list):
    """
    Constructs the full joint state (Idler + Return + Env) for one MC sample.
    Input Logic: Implicitly assumes Bell State input |Psi> = 1/sqrt(M) Sum |e_k>|e_k>
    """
    M = len(v0_list)
    psi_out = {} # Key: (k, (nS1, nE1, nS2, nE2...))

    for k in range(M):
        # Generate the Return+Env part for branch k
        phi_k = build_phi_k(k, v0_list, v1_list)
        
        # Combine with Idler state |e_k>_I (represented simply by index k)
        for se_state, amp in phi_k.items():
            key = (k, se_state) 
            # Apply 1/sqrt(M) normalization from the input Bell state
            psi_out[key] = psi_out.get(key, 0) + amp / np.sqrt(M)
    return psi_out

def pure_state_to_density_matrix(psi):
    rho = {}
    for k1, a1 in psi.items():
        for k2, a2 in psi.items():
            rho[(k1, k2)] = a1 * np.conjugate(a2)
    return rho

def trace_environment(rho):
    """
    Traces out environment modes (indices 1, 3, 5... in the tuple).
    Keeps Return modes (indices 0, 2, 4...).
    Idler is index 'k'.
    """
    rho_reduced = {}
    for (bra, ket), amp in rho.items():
        (ki, se_i) = bra
        (kj, se_j) = ket

        # se_i is (nS1, nE1, nS2, nE2 ...)
        # We keep evens (Signal/Return), check odds (Environment)
        signal_i = se_i[::2]
        env_i    = se_i[1::2]

        signal_j = se_j[::2]
        env_j    = se_j[1::2]

        if env_i == env_j: # Orthogonality of Fock states
            key = ((ki, signal_i), (kj, signal_j))
            rho_reduced[key] = rho_reduced.get(key, 0) + amp
    return rho_reduced


def monte_carlo_average(M, eta, nbar, Nmax, K):
    rho_final = {}
    print(f"Starting Monte Carlo ({K} samples)...")
    
    for i in range(K):
        n_vec = sample_environment(M, nbar, Nmax)
        v0_list, v1_list = local_bs_data(n_vec, eta, Nmax)
        psi_out = build_output_state(v0_list, v1_list)
        rho = pure_state_to_density_matrix(psi_out)
        rho_reduced = trace_environment(rho)
        
        for key, val in rho_reduced.items():
            rho_final[key] = rho_final.get(key, 0) + val
            
    # Average
    for key in rho_final:
        rho_final[key] /= K
        
    return rho_final


def get_basis_index(mode_counts, d):
    """Converts a tuple of photon counts (n1, n2...) to a flat index."""
    idx = 0
    stride = 1
    # Standard lexicographical mapping or big-endian
    # Here we use: n1*d^(M-1) + n2*d^(M-2)... 
    M = len(mode_counts)
    for i, n in enumerate(mode_counts):
        power = M - 1 - i
        idx += n * (d**power)
    return idx


def get_idler_basis_vector(k, M, d):
    """
    Returns the photon count tuple for |e_k>_I.
    |e_k> has 1 photon in mode k, 0 elsewhere.
    """
    counts = [0]*M
    counts[k] = 1
    return tuple(counts)

def sparse_to_dense(rho_dict, M, Nmax):
    """
    Converts the sparse dictionary to a NumPy array.
    Basis order: Idler (x) Return.
    """
    d = Nmax + 1
    dim_idler= M
    dim_return = d**M
    total_dim = dim_idler * dim_return
    
    mat = np.zeros((total_dim, total_dim), dtype=complex)
    
    for (key_bra, key_ket), val in rho_dict.items():
        (ki, signals_i) = key_bra
        (kj, signals_j) = key_ket
        
        
        idx_I_bra = ki
        idx_I_ket = kj
        
        # 2. Map Return Signals to Basis Index
        idx_R_bra = get_basis_index(signals_i, d)
        idx_R_ket = get_basis_index(signals_j, d)
        
        # 3. Combine for Full Index (Idler (x) Return)
        # Row = Idler_Bra * dim_R + Return_Bra
        row = idx_I_bra * dim_return + idx_R_bra
        col = idx_I_ket * dim_return + idx_R_ket
        
        mat[row, col] = val
        
    return mat

def check_correlations(rho_joint, dims):
    """
    Verifies if rho_joint is a product state (uncorrelated).
    dims = [dim_idler, dim_return]
    """
    print("\n--- Correlation Check ---")
    
    # 1. Calculate Marginal for Idler (Trace out Return - index 1)
    rho_I_marginal = partial_trace(rho_joint, dims, keep=[0])
    
    # 2. Calculate Marginal for Return (Trace out Idler - index 0)
    rho_R_marginal = partial_trace(rho_joint, dims, keep=[1])
    
    # 3. Reconstruct the hypothetical product state
    rho_product = np.kron(rho_I_marginal, rho_R_marginal)
    
    # 4. Compare
    diff_matrix = rho_joint - rho_product
    distance = np.linalg.norm(diff_matrix) # Frobenius norm
    
    print(f"Distance between Joint State and Product State: {distance:.10e}")
    
    if distance < 1e-9:
        print("RESULT: No Correlations detected. (State is separable/product).")
        return True
    else:
        print("RESULT: Correlations detected!")
        return False


def total_photons(index, M, d):
    photons = 0
    for _ in range(M):
        photons += index % d
        index //= d
    return photons


def block_diagonalize(rho, M, Nmax):
    d = Nmax + 1
    dim_idler = M
    dim_return = d**M
    
    Nc_values = []

    for i in range(rho.shape[0]):
        idler = i // dim_return
        ret_index = i % dim_return
        Nc = total_photons(ret_index, M, d)
        Nc_values.append(Nc)

    blocks = {}

    for Nc in sorted(set(Nc_values)):
        idx = [i for i,n in enumerate(Nc_values) if n==Nc]
        blocks[Nc] = rho[np.ix_(idx,idx)]

    return blocks



if __name__ == "__main__":
    # --- Parameters ---
    M = 2
    Nmax = 2
    Nbar = 0.5
    Kappa = 0.05  # Reflectivity (Target Strength)
    K_samples = 1000 # Number of Monte Carlo samples
    dim_idler = M
    dim_return = (Nmax+1)**M
    dim = dim_idler * dim_return
    d= Nmax+1
    
    print(f"Simulating Target Present (kappa={Kappa})...")
    
    # 1. Run Monte Carlo Simulation
    t0= time.time()
    rho_sparse = monte_carlo_average(M, Kappa, Nbar, Nmax, K_samples)
    
    # 2. Convert to Dense Matrix
    print("Converting to Dense Matrix...")
    rho_dense = sparse_to_dense(rho_sparse, M, Nmax)
    blocks= block_diagonalize(rho_dense, M, Nmax)
    t1= time.time()
    
    print(f"Done. Matrix Shape: {rho_dense.shape}\n")
    print(rho_dense)
    #print(np.real(np.diag(rho_dense)))
    print(blocks)
    print(f"The time it took to create the block diagonal matrix is : {t1-t0}")


    Nc = np.zeros((dim, dim))

    for i in range(dim):

        idler = i // dim_return
        ret_index = i % dim_return

        photons = total_photons(ret_index, M, d)

        Nc[i,i] = photons
    
    comm = rho_dense @ Nc - Nc @ rho_dense

    #print(f"The commutator relationship of N_c and the density matrix is given by: {comm}")
    
    # 3. Save
    np.save("rho_target_present.npy", rho_dense)
    
    # Optional: Check Trace
    total_trace=0
    for Nc, matrix in blocks.items():
        block_trace = np.real(np.trace(matrix))
        total_trace += block_trace
        print(f"\nBlock Nc = {Nc} | Shape: {matrix.shape} | Trace contribution: {block_trace:.4f}")
        print(np.round(matrix, 5)) # Uncomment to view matrices
        
    print(f"\nTotal Trace across all blocks: {total_trace:.4f} (Should be 1.0)")
    is_uncorrelated = check_correlations(rho_dense, [dim_idler, dim_return])
    print("Result: State is uncorrelated" if is_uncorrelated else "Result: State is correlated")