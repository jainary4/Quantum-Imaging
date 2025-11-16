"""

- Builds a Bell-like qudit state for M modes (idler + signal).
- Adds M environment ancilla qudits initialized in thermal states (truncated Fock).
- Demonstrates building the same circuit using MQT's QuantumCircuit (circuit construction only).
- Performs a full NumPy matrix simulation applying per-pair SWAP unitaries (eta=0 case).
- Partial traces out environment and verifies the result equals the expected target-absent state.

"""

import numpy as np
from scipy.linalg import expm
from Photon_number_Bell_State import create_bell_state_mqt
from mqt.qudits.simulation import MQTQuditProvider



# ---------------------------
# Helper linear-algebra utilities (NumPy)
# ---------------------------

def a_operator(d):
    """Annihilation operator a in truncated Fock basis |0..d-1> (d = Nmax+1)."""
    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n-1, n] = np.sqrt(n)
    return a

def swap_two_mode_matrix(d, include_phase=False):
    """
    Construct the d^2 x d^2 SWAP matrix for two truncated Fock modes.
    Maps |n>_A |m>_B --> |m>_A |n>_B (optionally with a phase).
    For our purposes the phase is irrelevant for number statistics; default phases=1.
    """
    D = d * d
    S = np.zeros((D, D), dtype=complex)
    for n in range(d):
        for m in range(d):
            src = n * d + m        # index for |n>_A |m>_B
            tgt = m * d + n        # index for |m>_A |n>_B
            phase = 1.0
            if include_phase:
                # example parity phase: (-1)^n (not necessary)
                phase = (-1.0) ** n
            S[tgt, src] = phase
    return S

def beam_splitter_unitary(d, eta):
    """
    Optional: Build general two-mode beam splitter U_BS(eta) via matrix exponential.
    We use generator G = theta (a1^\dagger a2 - a1 a2^\dagger) with cos(theta)=sqrt(eta).
    Returns a matrix of shape (d^2, d^2).
    """
    a = a_operator(d)
    I = np.eye(d, dtype=complex)
    a1 = np.kron(a, I)
    a2 = np.kron(I, a)
    sqrt_eta = np.sqrt(max(0.0, eta))
    sqrt_1_eta = np.sqrt(max(0.0, 1.0 - eta))
    # choose theta so cos(theta)=sqrt_eta, sin(theta)=sqrt(1-eta)
    theta = np.arctan2(sqrt_1_eta, sqrt_eta)
    G = theta * (a1.conj().T @ a2 - a1 @ a2.conj().T)
    U = expm(G)
    return U

def kron_list(matrices):
    """Kronecker product of list of matrices in order."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

def partial_trace(rho, dims, keep):
    """
    Partial trace.
    - rho: (D,D) density matrix where D = product(dims)
    - dims: list of subsystem dimensions (e.g. [dim_I, dim_S, dim_E])
    - keep: list of subsystem indices to keep (0-based)
    Returns reduced density matrix on the kept subsystems.
    """
    dims = list(dims)
    N = len(dims)
    assert rho.shape == (int(np.prod(dims)), int(np.prod(dims)))
    # reshape to (i1,i2,...,iN, j1,j2,...,jN)
    reshaped = rho.reshape(dims + dims)
    # trace over subsystems not in keep
    trace_out = [i for i in range(N) if i not in keep]
    # Do traces in descending order so earlier indices aren't shifted
    for t in sorted(trace_out, reverse=True):
        reshaped = np.trace(reshaped, axis1=t, axis2=t + N)
    # Now reshaped has shape (dims_keep..., dims_keep...)
    keep_dims = [dims[i] for i in keep]
    final = reshaped.reshape((int(np.prod(keep_dims)), int(np.prod(keep_dims))))
    return final

# ---------------------------
# Physics helpers: thermal state, single-photon pattern, Bell rho
# ---------------------------

def thermal_rho_single(d, Nbar):
    """
    Single-mode truncated thermal density matrix (d levels).
    P_n = (Nbar^n) / (1+Nbar)^{n+1}, renormalized over truncation.
    """
    n = np.arange(d)
    P = (Nbar ** n) / ((1 + Nbar) ** (n + 1))
    P = P / np.sum(P)   # renormalize if truncated
    return np.diag(P)



def build_bell_rho_IS(M, d):
    """
    Build ρ_IS from the MQT Bell-state circuit returned by create_bell_state_mqt.
    Assumes create_bell_state_mqt is already imported and works.
    """
    circuit = create_bell_state_mqt(M, d)

    dim_bank = d ** M
    total_dim = dim_bank * dim_bank
    
    # Use simulation backend
    provider = MQTQuditProvider()
    backend = provider.get_backend("tnsim")  # Tensor network simulator
    
    # Run circuit
    job = backend.run(circuit)
    result = job.result()
    
    # Get state vector
    psi = result.get_state_vector()
    psi = np.array(psi)
    psi /= np.linalg.norm(psi)
    
    return np.outer(psi, psi.conj())




def simulate_unitary_swap_target_absent(M=2, Nmax=5, Nbar=0.5, include_phase=False):
    """
    Numeric simulation:
      - Build rho_IS (Bell)
      - Build rho_E = tensor of single-mode thermal states (each d = Nmax+1)
      - Build per-pair SWAP unitaries (d^2 x d^2). Global U_SE = kron(S1, S2, ...)
      - Build U_full = I_I ⊗ U_SE and apply to rho_total = rho_IS ⊗ rho_E
      - Partial trace over environment to obtain rho_out
      - Construct expected rho_expected = Tr_S[rho_IS] ⊗ rho_E and compare
    """
    d = Nmax + 1
    dim_bank = d ** M   # dimension of each bank (I or S or E)
    print(f"Simulate: M={M}, Nmax={Nmax} (d={d}), dim_bank={dim_bank}")

    # 1) build rho_IS
    rho_IS = build_bell_rho_IS(M, d)            # shape (dim_bank^2, dim_bank^2)

    # 2) build rho_E = tensor product of M single-mode thermal states
    rho_B_single = thermal_rho_single(d, Nbar)
    rho_E = rho_B_single
    for _ in range(1, M):
        rho_E = np.kron(rho_E, rho_B_single)
    dim_E = rho_E.shape[0]
    assert dim_E == dim_bank

    # 3) full input rho_total = rho_IS ⊗ rho_E
    rho_total = np.kron(rho_IS, rho_E)
    # dims ordering: I (dim_bank) ⊗ S (dim_bank) ⊗ E (dim_bank)

    # 4) build per-pair SWAP unitaries and global U_SE (acts on S⊗E)
    S_single = swap_two_mode_matrix(d, include_phase=include_phase)  # d^2 x d^2
    # global U_SE is (S1 ⊗ S2 ⊗ ... ⊗ SM), ordering (S1..SM,E1..EM) is baked in how we kronecker
    U_SE = S_single
    for k in range(1, M):
        U_SE = np.kron(U_SE, S_single)
    # U_full = I_I ⊗ U_SE
    I_I = np.eye(dim_bank, dtype=complex)
    U_full = np.kron(I_I, U_SE)   # acts on I ⊗ (S ⊗ E)

    # 5) evolve
    rho_after = U_full @ rho_total @ U_full.conj().T

    # 6) partial trace over environment (keep [I, S] -> subsystems 0 and 1)
    dims = [dim_bank, dim_bank, dim_bank]  # I, S, E
    rho_out = partial_trace(rho_after, dims, keep=[0, 1])

    # 7) expected: Tr_S[rho_IS] ⊗ rho_E
    # Compute Tr_S[rho_IS] (trace over second bank)
    rho_idler_reduced = partial_trace(rho_IS, [dim_bank, dim_bank], keep=[0])
    rho_expected = np.kron(rho_idler_reduced, rho_E)

    # Diagnostics
    trace_out = np.trace(rho_out)
    trace_expected = np.trace(rho_expected)
    norm_diff = np.linalg.norm(rho_out - rho_expected)

    print(f"Tr(rho_out) = {trace_out:.12f}, Tr(rho_expected) = {trace_expected:.12f}")
    print(f"|| rho_out - rho_expected ||_F = {norm_diff:.6e}")
    return {
        'rho_out': rho_out,
        'rho_expected': rho_expected,
        'rho_after': rho_after,
        'rho_total': rho_total,
        'rho_IS': rho_IS,
        'rho_E': rho_E
    }



if __name__ == "__main__":
    # Parameters you can change
    M = 2
    Nmax = 2       # truncation: max photon number per mode
    Nbar = 0.5      # mean photon number for thermal environment
    eta = 0.0       # we implement eta=0 by using SWAP unitaries
    include_phase = False

    data = simulate_unitary_swap_target_absent(M=M, Nmax=Nmax, Nbar=Nbar, include_phase=include_phase)

    # If norm_diff is tiny, we succeeded
    rho_out = data['rho_out']
    rho_expected = data['rho_expected']
    print("\nSanity check (first few diagonal elements of rho_out):")
    print(np.real(np.diag(rho_out)))
    print("\nSanity check (first few diagonal elements of rho_expected):")
    print(np.real(np.diag(rho_expected)))

    # Save final rho_out to file if desired:
    np.save("rho_out.npy", rho_out)
