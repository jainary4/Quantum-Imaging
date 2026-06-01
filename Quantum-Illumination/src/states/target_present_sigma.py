import modal
import numpy as np
import time
from scipy.linalg import expm
from pathlib import Path
from typing import Dict, List, Tuple
import json
import itertools
app = modal.App("quantum-illumination")

# Use a light CPU‑only image – no GPU needed now.
cpu_image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install("numpy", "scipy")
)

@app.function(
    image=cpu_image,
    timeout=3000,
    cpu=1,                               # 4 CPU cores are plenty
    volumes={"/vol/qi": modal.Volume.from_name("qi-results", create_if_missing=True)}
)
def run_full_pipeline(M: int,Kappa: float,Nbar: float,Nmax: int,K_samples: int,exact:bool=False) -> dict:
    """
    Generate the averaged sigma matrices for both hypotheses.

    Parameters
    ----------
    M : int
        Number of mode pairs.
    Kappa : float
        Target reflectivity (beam‑splitter parameter η).
    Nbar : float
        Mean thermal photon number per environment mode.
    Nmax : int
        Fock‑space truncation (max photons per mode).
    K_samples : int
        Number of Monte Carlo samples.

    Returns
    -------
    metadata : dict
        Dictionary with run parameters and summary.
    """

    def bose_einstein_pmf(n: int, nbar: float) -> float:
        """
        Bose‑Einstein probability mass function for a single thermal mode.

        Parameters
        ----------
        n : int
            Number of photons in the mode.
        nbar : float
            Mean photon number of the thermal state.

        Returns
        -------
        float
            Probability P(n) = (nbar^n) / (nbar+1)^(n+1).
        """
        return (nbar**n) / ((nbar + 1)**(n + 1))



    def thermal_distribution(nbar: float, Nmax: int) -> np.ndarray:
        """
        Normalised thermal photon‑number distribution truncated to [0, Nmax].

        Parameters
        ----------
        nbar : float
            Mean photon number.
        Nmax : int
            Maximum photon number retained (Fock truncation).

        Returns
        -------
        probs : ndarray of shape (Nmax+1,)
            Probability vector where probs[n] = P(n) for n = 0 … Nmax.
            The vector sums to 1.
        """
        probs = np.array([bose_einstein_pmf(n, nbar) for n in range(Nmax + 1)])
        probs /= probs.sum()
        return probs


    def sample_environment(M: int, nbar: float, Nmax: int) -> np.ndarray:
        """
        Draw one configuration of thermal noise for all M environment modes.

        Parameters
        ----------
        M : int
            Number of spatial/spectral modes.
        nbar : float
            Mean photon number in each environment mode.
        Nmax : int
            Maximum photon number per mode (truncation).

        Returns
        -------
        n_vec : ndarray of shape (M,), dtype int
            Photon numbers (n0, n1, ..., n_{M-1}) in the M environment modes,
            sampled independently from the thermal distribution.
        """
        probs = thermal_distribution(nbar, Nmax)
        return np.random.choice(np.arange(Nmax + 1), size=M, p=probs)




    def beam_splitter_fock(nS: int, nE: int, eta: float, Nmax: int) -> Dict[Tuple[int, int], complex]:
        """
        Compute the exact output state of a beam splitter acting on Fock states.

        Parameters
        ----------
        nS : int
            Number of photons in the signal input port (0 or 1 in this protocol).
        nE : int
            Number of photons in the environment input port.
        eta : float
            Beam‑splitter reflectivity (target strength), 0 ≤ η ≤ 1.
        Nmax : int
            Global photon‑number cutoff; output components with nS_out > Nmax
            or nE_out > Nmax are discarded.

        Returns
        -------
        amps : dict
            Dictionary mapping (nS_out, nE_out) → complex amplitude.
            Only components with non‑negligible amplitude (> 1e‑12) are kept.
        """
        amps = {}
        N = nS + nE
        dim = N + 1

        # Generator G for the beam‑splitter interaction in the fixed‑N subspace.
        G = np.zeros((dim, dim))
        for k in range(N):
            val = np.sqrt(k + 1) * np.sqrt(N - k)
            G[k + 1, k] = val
            G[k, k + 1] = -val

        theta = np.arccos(np.sqrt(eta))
        U = expm(theta * G)     # unitary evolution matrix

        input_idx = nS          # column corresponding to |nS⟩_S |nE⟩_E
        for k_out in range(dim):
            amp = U[k_out, input_idx]
            nS_out = k_out
            nE_out = N - k_out
            if nS_out <= Nmax and nE_out <= Nmax:
                if abs(amp) > 1e-12:
                    amps[(nS_out, nE_out)] = amp
        return amps



    def compute_v_states(n_env: int, eta: float, Nmax: int) -> Tuple[Dict, Dict]:
        """
        Compute the two possible local output states for a given environment photon count.

        Parameters
        ----------
        n_env : int
            Photon number in the environment mode.
        eta : float
            Beam‑splitter reflectivity.
        Nmax : int
            Fock‑space truncation.

        Returns
        -------
        v0 : dict {(nS, nE): amplitude}
            Output when the signal mode contained 0 photons (idler photon absent).
        v1 : dict {(nS, nE): amplitude}
            Output when the signal mode contained 1 photon (idler photon present).
        """
        v0 = beam_splitter_fock(nS=0, nE=n_env, eta=eta, Nmax=Nmax)
        v1 = beam_splitter_fock(nS=1, nE=n_env, eta=eta, Nmax=Nmax)
        return v0, v1



    def local_bs_data(n_vec: np.ndarray, eta: float, Nmax: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply the beam‑splitter interaction to every mode for one environment sample.

        Parameters
        ----------
        n_vec : ndarray of shape (M,), dtype int
            Photon numbers in the M environment modes.
        eta : float
            Beam‑splitter reflectivity.
        Nmax : int
            Fock‑space truncation.

        Returns
        -------
        v0_list : list of M dicts
            Each dict is the (vacuum, environment) output for one mode.
        v1_list : list of M dicts
            Each dict is the (single‑photon, environment) output for one mode.
        """
        v0_list = []
        v1_list = []
        for n_j in n_vec:
            v0, v1 = compute_v_states(n_j, eta, Nmax)
            v0_list.append(v0)
            v1_list.append(v1)
        return v0_list, v1_list
       

    
    def compute_mode_sigmas(v0_dict: Dict[Tuple[int, int], complex],v1_dict: Dict[Tuple[int, int], complex],d: int) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Convert local beam‑splitter output states for one mode into the four
        reduced sigma matrices (partial traces over the environment).

        Parameters
        ----------
        v0_dict : dict {(s, e): complex}
            Output amplitudes when the signal mode contained 0 photons.
            Keys are (nS_out, nE_out) tuples; values are complex amplitudes.
        v1_dict : dict {(s, e): complex}
            Output amplitudes when the signal mode contained 1 photon.
        d : int
            Dimension of the mode's Fock space (Nmax + 1).

        Returns
        -------
        sigmas : dict
            Dictionary containing four CuPy arrays (all shape (d, d)):
            (0,0) → sigma for (q'=0, q=0)   ⟨v0|·⟩⟨·|v0⟩?
            (1,1) → sigma for (q'=1, q=1)   ... etc.
            (0,1) → sigma for (q'=0, q=1)   ...
            (1,0) → sigma for (q'=1, q=0)   ...
            Each sigma matrix is computed as v_q @ v_q'^H, which corresponds to
            tracing out the environment degree of freedom:
                σ^{(q',q)}_ij = Σ_e (v_q)_{i,e} (v_{q'})_{j,e}^*.
        """
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
    
    def compute_total_trace_from_sigmas(sigma: np.ndarray, M: int, Nmax: int) -> float:

        """Compute Tr[ρ] from the sigma matrices."""
        d = Nmax + 1
        trace = 0.0
        for k in range(M):
            for r_tuple in itertools.product(range(d), repeat=M):
                prod = 1.0 + 0.0j
                for m in range(M):
                    q = 1 if m == k else 0
                    prod *= sigma[m, q, q, r_tuple[m], r_tuple[m]]
                trace += np.real(prod) / M
        return trace

    d = Nmax + 1
    sigma_pres = np.zeros((M, 2, 2, d, d), dtype=complex)

    t0 = time.time()
    if exact:
        # ----- Exact enumeration over all environment configurations -----
        probs = thermal_distribution(Nbar, Nmax)
        all_env_tuples = list(itertools.product(range(d), repeat=M))
        env_probs = [np.prod([probs[n] for n in vec]) for vec in all_env_tuples]

        for n_vec, p in zip(all_env_tuples, env_probs):
            v0_list, v1_list = local_bs_data(np.array(n_vec), Kappa, Nmax)
            for j in range(M):
                sigmas = compute_mode_sigmas(v0_list[j], v1_list[j], d)
                sigma_pres[j, 0, 0] += p * sigmas[(0,0)]
                sigma_pres[j, 1, 1] += p * sigmas[(1,1)]
                sigma_pres[j, 0, 1] += p * sigmas[(0,1)]
                sigma_pres[j, 1, 0] += p * sigmas[(1,0)]
    else:
        # ----- Monte‑Carlo sampling -----
        for _ in range(K_samples):
            n_vec = sample_environment(M, Nbar, Nmax)
            v0_list, v1_list = local_bs_data(n_vec, Kappa, Nmax)
            for j in range(M):
                sigmas = compute_mode_sigmas(v0_list[j], v1_list[j], d)
                sigma_pres[j, 0, 0] += sigmas[(0,0)]
                sigma_pres[j, 1, 1] += sigmas[(1,1)]
                sigma_pres[j, 0, 1] += sigmas[(0,1)]
                sigma_pres[j, 1, 0] += sigmas[(1,0)]
        sigma_pres /= K_samples

    t1 = time.time()
    print(f"Sigma matrices accumulated in {t1 - t0:.2f} s")

    T = compute_total_trace_from_sigmas(sigma_pres, M, Nmax)
    sigma_pres /= T**(1.0 / M)
    result= compute_total_trace_from_sigmas(sigma_pres,M,Nmax)
    print(f"the trace of normalised sigma pres is : {result}") 


    if exact:
        K_val="exact"
    else:
        K_val= K_samples
        
    run_dir = Path("/vol/qi") / f"M={M}_K={K_val}_Nbar={Nbar}_Nmax={Nmax}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "sigmas.npy", sigma_pres)


    print(f"Saved sigma matrices to {run_dir}")
    return 


@app.local_entrypoint()
def main(
    m: int = 15,
    kappa: float = 0.05,
    nbar: float = 0.5,
    nmax: int = 2,
    k_samples: int = 1000000
):
    print(f"Running with M={m}, Kappa={kappa}, Nbar={nbar}, Nmax={nmax}, K_samples={k_samples}")
    run_full_pipeline.remote(m, kappa, nbar, nmax, k_samples,exact= False)
  