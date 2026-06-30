from POVM_interferometric import apply_slm, build_local_BS_table,generate_patterns_for_input_state,generate_input_states,chunkify, process_chunk, build_pattern_amplitudes_parallel, build_pattern_amplitudes_serial,convert_lookup_to_json_serializable, download_file
import numpy as np
from typing import Dict, List , Tuple, Iterator,Any
import math
import itertools
from itertools import product
from multiprocessing import Pool
from collections import defaultdict
import json
from pathlib import Path
import modal
import tempfile
import asyncio
import subprocess
import pickle
import time

def compute_probabilities_from_sigmas(
    entries: List[Tuple[int, float]],          # (idx, amplitude)
    basis_list: List[Tuple[int, Tuple[int, ...]]],  # canonical order for this Nc
    sigma_pres: np.ndarray,                    # shape (M, 2, 2, d, d)
    sigma_abs: np.ndarray,                     # shape (M, 2, 2, d, d)
    M: int,
    norm_pres: float,
    norm_abs: float,
    phases: np.ndarray = None,                 # optional SLM phases, length M
    phase_R: np.ndarray = None                 # optional per-mode return phases (if needed)
) -> Tuple[float, float, float]:
    """
    Compute p0, p1, Λ using the factorised sigma matrices.

    For each pair (i,j) of non‑zero amplitude indices,
        ρ_{ij} = (1/M) ∏_m σ_m^{(q'_m, q_m)}(r'_m, r_m)

    where (k_i, r_i) is the i‑th basis state, and similarly for j.
    If SLM phases are given, multiply each term by exp(i(θ_i - θ_j)).
    """
    # Pre‑compute the total phase per basis state if SLM is on
    n_entries = len(entries)
    theta = np.zeros(n_entries) if phases is not None else None

    if phases is not None:
        # phases: idler phases only, we assume return phases are zero for now.
        # (You can extend to include return phases if desired.)
        for idx_in_list, (global_idx, _) in enumerate(entries):
            k_i, r_i = basis_list[global_idx]
            # Phase: φ_I(k_i) + sum over modes of φ_R(m)*r_i[m] (if φ_R given)
            theta_i = phases[k_i]
            if phase_R is not None:
                theta_i += np.dot(phase_R, r_i)
            theta[idx_in_list] = theta_i

    p0 = 0.0
    p1 = 0.0

    # Double loop over the (sparse) entries
    for i_idx, (global_i, a_i) in enumerate(entries):
        k_i, r_i = basis_list[global_i]
        for j_idx, (global_j, a_j) in enumerate(entries):
            k_j, r_j = basis_list[global_j]

            # Compute product of sigma scalars
            prod_pres = 1.0 + 0.0j
            prod_abs  = 1.0 + 0.0j
            for m in range(M):
                q_prime = 1 if m == k_i else 0
                q       = 1 if m == k_j else 0
                r_prime = r_i[m]
                r       = r_j[m]
                prod_pres *= sigma_pres[m, q_prime, q, r_prime, r]
                prod_abs  *= sigma_abs[m, q_prime, q, r_prime, r]

            # Multiply by 1/M (the normalisation from the Bell state)
            rho_ij_pres = prod_pres / (M )
            rho_ij_abs  = prod_abs  / (M )

            # SLM phase factor
            if phases is not None:
                phase_factor = np.exp(1j * (theta[i_idx] - theta[j_idx]))
                rho_ij_pres *= phase_factor
                rho_ij_abs  *= phase_factor

            # Accumulate: term = conj(a_i) * a_j * rho_ij
            p0 += np.real(np.conj(a_i) * a_j * rho_ij_abs)
            p1 += np.real(np.conj(a_i) * a_j * rho_ij_pres)

    if p0 <= 0 or p1 <= 0:
        lam = 0.0
    else:
        lam = np.log(p1 / p0)

    return lam, p0, p1


def process_one_block_sigmas(args: Tuple) -> Tuple[int, Dict[Tuple[int, ...], Tuple[float, float, float]]]:
    """
    Worker function for one Nc block using sigma matrices.

    Input args: (M, Nc, local_BS, N_max, sigma_pres, sigma_abs,
                 basis_list, inner_workers, use_slm, phases)
    Returns (Nc, lookup_table_for_this_block).
    """
    (M, Nc, local_BS, N_max, sigma_pres, sigma_abs,
     basis_list, inner_workers, use_slm, phases,
     norm_pres, norm_abs) = args

    # Build amplitude dictionary – parallel or serial
    if inner_workers > 1:
        pattern_amps = build_pattern_amplitudes_parallel(M, Nc, local_BS, N_max, num_workers=inner_workers)
    else:
        pattern_amps = build_pattern_amplitudes_serial(M, Nc, local_BS, N_max)

    # Compute probabilities and Λ for every output pattern
    block_table = {}
    for pattern, entries in pattern_amps.items():
        lam, p0, p1 = compute_probabilities_from_sigmas(
            entries, basis_list, sigma_pres, sigma_abs, M, norm_pres, norm_abs,
            phases=phases if use_slm else None
        )
        block_table[pattern] = (lam, p0, p1)

    return Nc, block_table

def compute_total_trace_from_sigmas(sigma: np.ndarray, M: int, Nmax: int) -> float:
    """
    Compute Tr[ρ] for the full idler‑return state defined by the sigma matrices.
    sigma shape : (M, 2, 2, d, d),   d = Nmax + 1.
    Returns the sum over all basis states (k, r) of
        (1/M) * ∏_m sigma[m, q_m, q_m, r_m, r_m]
    """
    d = Nmax + 1
    trace = 0.0
    # Iterate over all idler modes and all possible return configurations
    for k in range(M):
        # Generate all return tuples with total photons 0..M*Nmax
        # We can use itertools.product to loop over all d^M configurations
        for r_tuple in itertools.product(range(d), repeat=M):
            prod = 1.0 + 0.0j
            for m in range(M):
                q = 1 if m == k else 0
                r = r_tuple[m]
                prod *= sigma[m, q, q, r, r]
            trace += np.real(prod) / M
    return trace

def build_global_lookup_table_sigmas(
    M: int,
    N_max: int,
    local_BS: dict,
    sigma_pres: np.ndarray,
    sigma_abs: np.ndarray,
    norm_pres, 
    norm_abs,
    Nc_list: List[int] = None,
    outer_workers: int = 1,
    inner_workers: int = 1,
    use_slm: bool = False,
    phases: np.ndarray = None
) -> Dict[int, Dict[Tuple[int, ...], Tuple[float, float, float]]]:
    """
    Build lookup table using sigma matrices instead of block matrices.
    """
    if Nc_list is None:
        Nc_list = list(range(M * N_max + 1))  # all possible Nc

    tasks = []
    for Nc in Nc_list:
        # Generate the canonical basis list for this Nc (indices match the amplitude dictionary)
        basis_list = generate_input_states(M, Nc, N_max)   # returns list of (k, r_tuple)
        if len(basis_list) == 0:
            continue
        tasks.append((
            M, Nc, local_BS, N_max, sigma_pres, sigma_abs,
            basis_list, inner_workers, use_slm, phases,norm_pres, norm_abs
        ))

    if outer_workers > 1:
        with Pool(outer_workers) as pool:
            results = pool.map(process_one_block_sigmas, tasks)
    else:
        results = [process_one_block_sigmas(task) for task in tasks]

    global_lookup = {}
    for Nc, block_table in results:
        global_lookup[Nc] = block_table
    return global_lookup



if __name__ == "__main__":
    # --- Parameters for the simulation ---
    M = 9
    Nmax = 2
    kappa = 0.05      # target reflectivity (used in target‑present generation)
    Nbar = 0.5        # thermal background
    samples=1000000

    VOLUME_NAME = "qi-results"
    volume = modal.Volume.from_name(VOLUME_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # remote paths (as before)
        pres_remote = f"M={M}_K=exact_Nbar={Nbar}_Nmax={Nmax}/sigmas.npy"
        abs_remote  = f"Target_Absent/M={M}_Nbar={Nbar}_Nmax={Nmax}/sigmas.npy"
        bs_remote   = f"local_BS_tables/local_BS_table_Nmax={Nmax}/local_BS_Nmax{Nmax}.pkl"

        # local file paths
        pres_local = tmp / "pres.npz"
        abs_local  = tmp / "abs.npz"
        bs_local   = tmp / "local_BS.pkl"

        # Download using Modal CLI
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, pres_remote, str(pres_local)], check=True)
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, abs_remote,  str(abs_local)],  check=True)
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, bs_remote,   str(bs_local)],   check=True)

        
       
        sigma_pres = np.load(str(pres_local))   # shape (M, 2, 2, d, d)
        sigma_abs  = np.load(str(abs_local))    # shape (M, 2, 2, d, d)

        T_pres = compute_total_trace_from_sigmas(sigma_pres, M, Nmax)
        T_abs  = compute_total_trace_from_sigmas(sigma_abs,  M, Nmax)

        print(f"norm_pres={T_pres:.8f}, norm_abs={T_abs:.8f}")

        with open(str(bs_local), "rb") as f:
            local_BS = pickle.load(f)
        
        Nc_list = list(range(M * Nmax + 1))
        start_time = time.time()
        global_lookup=  build_global_lookup_table_sigmas(M=M,
            N_max=Nmax,
            local_BS=local_BS,
            sigma_pres=sigma_pres,
            sigma_abs=sigma_abs,
            norm_pres=T_pres, 
            norm_abs=T_abs,
            Nc_list=Nc_list,
            outer_workers=1,   # adjust for your device/specs
            inner_workers=10,
            use_slm=False,
        )

        end_time= time.time()
        print(f"Time for M={M} and Nmax= {Nmax} is: {end_time - start_time} seconds ")


        pickle_path = tmp / "lookup_sigma.pkl"

        with open(str(pickle_path), "wb") as f:
            pickle.dump(global_lookup, f)
        
        json_serializable = convert_lookup_to_json_serializable(global_lookup)
        json_path = tmp / "lookup_sigma.json"

        with open(str(json_path), "w") as f:
            json.dump(json_serializable, f, indent=2)
        
        remote_dir = f"Global_lookuptable/M={M}_Nmax={Nmax}"
        print(f"Uploading lookup table to volume at {remote_dir}/ ...")
        with volume.batch_upload() as batch:
            batch.put_file(str(pickle_path), f"{remote_dir}/lookup_sigma_exact.pkl")
            batch.put_file(str(json_path), f"{remote_dir}/lookup_sigma_exact.json")