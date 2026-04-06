import numpy as np
from scipy.linalg import expm

def a_operator(d):
    """Annihilation operator a in truncated Fock basis."""
    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n-1, n] = np.sqrt(n)
    return a

def swap_two_mode_matrix(d, include_phase=False):
    """Construct the d^2 x d^2 SWAP matrix."""
    D = d * d
    S = np.zeros((D, D), dtype=complex)
    for n in range(d):
        for m in range(d):
            src = n * d + m
            tgt = m * d + n
            S[tgt, src] = 1.0
    return S

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

def thermal_rho_single(d, Nbar):
    """Single-mode truncated thermal density matrix."""
    n = np.arange(d)
    if Nbar == 0:
        P = np.zeros(d); P[0]=1.0
    else:
        P = (Nbar ** n) / ((1 + Nbar) ** (n + 1))
        P = P / np.sum(P) # Renormalize
    return np.diag(P)



def get_bell_state_rho(M, d):
    """
    Generates the Bell State Density Matrix using pure NumPy.
    State: |Psi> = (1/sqrt(M)) * Sum(|e_k>_I |e_k>_S) for k=1..M
    """
    dim_bank = d**M
    dim_total = dim_bank * dim_bank
    
    # We construct the state vector |Psi>
    psi_vector = np.zeros(dim_total, dtype=complex)
    
    # Basis vectors
    vac = np.zeros(d); vac[0] = 1.0
    one = np.zeros(d); one[1] = 1.0
    
    for k in range(M):
        # Build |e_k> : 1 photon in k-th mode, 0 in others
        factors = []
        for m in range(M):
            factors.append(one if m == k else vac)
        
        # Tensor product for Idler part
        vec_k = factors[0]
        for f in factors[1:]:
            vec_k = np.kron(vec_k, f)
            
        # For Bell state, Signal is same state as Idler: |e_k>_I |e_k>_S
        term = np.kron(vec_k, vec_k) 
        psi_vector += term

    # Normalize
    psi_vector /= np.linalg.norm(psi_vector)
    
    # Convert to Density Matrix
    return np.outer(psi_vector, psi_vector.conj())



def simulate_target_absent(M, Nmax, Nbar):
    """
    Simulates Target Absent Case.
    Since Transmissivity = 0, the output is simply:
    Idler (from Bell State) (tensor) Thermal Noise (from Environment)
    """
    d = Nmax + 1
    dim_bank = d ** M
    
    print(f"Generating Target Absent State (M={M}, Nmax={Nmax})...")

    rho_idler = np.eye(M) / M
    
    # 3. Generate Thermal Noise for Return path
    # (Since target is absent, what we receive is just noise)
    rho_single_therm = thermal_rho_single(d, Nbar)
    rho_return = rho_single_therm
    for _ in range(1, M):
        rho_return = np.kron(rho_return, rho_single_therm)
        
    # 4. Combine: Output = Idler (x) Return_Noise
    rho_final = np.kron(rho_idler, rho_return)
    
    return rho_final

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


if __name__ == "__main__":
    # --- INPUTS ---
    M = 2              # Number of mode pairs
    Nmax = 2           # Max photons per mode
    Nbar = 0.5         # Noise brightness
    
    # Run Generation
    # Note: I replaced the slow "swap simulation" with the mathematically 
    # equivalent and much faster analytical construction in 'simulate_target_absent'
    rho_out = simulate_target_absent(M, Nmax, Nbar)

    print("\nSanity check (first few diagonal elements):")
    print(np.real(np.diag(rho_out)))
    
    # Save for POVM step
    np.save("rho_target_absent.npy", rho_out)
    print("\nState saved to 'rho_target_absent.npy'")
    print(f"Shape of rho_out: {rho_out.shape}")
    print(rho_out)
    dim_bank = (Nmax+1)**M 
    is_uncorrelated = check_correlations(rho_out, [M, dim_bank])
    print("Result: State is uncorrelated" if is_uncorrelated else "Result: State is correlated")
    
