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

"""We are building the interferometric POVM for the target present and absent states."""

def apply_slm(block_matrix, k_array, phases ):
    """
    Apply the spatial light modulator (SLM) phases to a block matrix.

    Parameters
    ----------
    block : ndarray of shape (D, D), complex
        The block diagonal matrix B for a given Nc.
    k_array : ndarray of shape (D,), int
        Idler mode index (0..M-1) for each basis state in the block.
    phases : ndarray of shape (M,), float
        Phase applied to each idler mode, in radians.

    Returns
    -------
    new_block : ndarray of shape (D, D), complex
        Block matrix after SLM transformation.
    """
    D = block_matrix.shape[0]
    # Compute phase factor for each basis state
    phase_factors = np.exp(1j * phases[k_array])   # shape (D,)
    # Outer difference: exp(i*(phi_bra - phi_ket))
    phase_matrix = phase_factors[:, None] * np.conj(phase_factors[None, :])
    # Element-wise multiplication with the block
    return block * phase_matrix


def build_local_BS_table(N_max: int) -> Dict[Tuple[int, int], List[Tuple[int, int, float]]]:
    """
    Precompute all non‑zero transition amplitudes for a single 50:50 beam splitter.

    The beam splitter acts on one idler mode (I) and one return mode (R).  
    Because the Bell‑state source emits at most one photon per idler mode,
    we only need to consider nI ∈ {0, 1}. The return mode can contain up to
    N_max photons, nR ∈ {0, 1, …, N_max}.

    For each input pair (nI, nR), the output states |nc, nd⟩ with  
    nc + nd = nI + nR are computed using the analytic formula:

        ⟨nc, nd| U_BS |nI, nR⟩ =
            √(nI! nR! nc! nd!) · (1/√2)^(nI+nR) ·
            Σ_{j=max(0,nc−nR)}^{min(nI,nc)} 
                (−1)^(nR − nc + j) / [j! (nI−j)! (nc−j)! (nR−nc+j)!]

    The result is stored as a dictionary:
        key   : (nI, nR)
        value : list of (nc, nd, amplitude) for all reachable output pairs
                where |amplitude| > 1e-15.

    Parameters
    ----------
    N_max : int
        Maximum photon number per return mode (Fock cutoff).

    Returns
    -------
    local_BS : dict
        Dictionary mapping (nI, nR) → list of (nc, nd, amplitude).
    """
    # Precompute factorials up to max needed: nI ≤ 1, nR ≤ N_max,
    # nc, nd ≤ N_max+1, so max index = N_max + 1.
    max_fact: int = N_max + 1
    factorial: List[int] = [math.factorial(i) for i in range(max_fact + 2)]
    
    # Initialize the lookup table dictionary
    local_BS: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = {}
    
    # Loop over all possible idler photon numbers (only 0 or 1)
    for nI in (0, 1):
        # Loop over all possible return photon numbers up to N_max
        for nR in range(N_max + 1):
            S: int = nI + nR                      # total photons in the pair
            scale: float = (1.0 / math.sqrt(2)) ** S   # (1/√2)^{nI+nR}
            sqrt_prefactor: float = math.sqrt(factorial[nI] * factorial[nR])

            out_list: List[Tuple[int, int, float]] = []
            # Output photon numbers must sum to S
            for nc in range(S + 1):
                nd: int = S - nc
                # --- Closed-form amplitude for |nI,nR⟩ → |nc,nd⟩ ---
                # Summation bounds
                j_min: int = max(0, nc - nR)
                j_max: int = min(nI, nc)

                term_sum: float = 0.0
                for j in range(j_min, j_max + 1):
                    # Denominator: j! (nI−j)! (nc−j)! (nR−nc+j)!
                    denom: int = (factorial[j] *
                                  factorial[nI - j] *
                                  factorial[nc - j] *
                                  factorial[nR - nc + j])
                    # Numerator sign: (−1)^{nR − nc + j}
                    sign: int = 1 if (nR - nc + j) % 2 == 0 else -1
                    term_sum += sign / denom

                # Full amplitude: sqrt(nI! nR! nc! nd!) * scale * Σ
                # Note: sqrt(nc! nd!) is multiplied inside the sum.
                amp: float = (sqrt_prefactor *
                             math.sqrt(factorial[nc] * factorial[nd]) *
                             scale *
                             term_sum)

                # Keep only non‑zero amplitudes (within numerical precision)
                if abs(amp) > 1e-15:
                    out_list.append((nc, nd, amp))

            local_BS[(nI, nR)] = out_list

    return local_BS


def generate_patterns_for_input_state(k: int, r_tuple: Tuple[int, ...], local_BS: dict, M: int) -> Iterator[Tuple[Tuple[int, ...], float]]:

    """
    Generate all reachable global output patterns for a single input state.

    Parameters
    ----------
    k : int
        The mode index that contains the idler photon (0 ≤ k < M).
    r_tuple : tuple of int, length M
        Return photon numbers per mode. Sum must equal N_C.
    local_BS : dict
        Precomputed local beam‑splitter amplitude table.
    M : int
        Number of mode pairs.

    Yields
    ------
    pattern : tuple of int (length 2M)
        Flat tuple (n_c0, n_d0, n_c1, n_d1, …).
    amplitude : float
        Amplitude of this output pattern.
    """
    # Step 1: Build list of per‑pair output options
    per_pair_options = []
    for j in range(M):
        nI = 1 if j == k else 0
        nR = r_tuple[j]
        options = local_BS[(nI, nR)]  # list of (nc, nd, amp)
        per_pair_options.append(options)
    
    # Step 2: Cartesian product over all pairs
    for combo in product(*per_pair_options):
        # combo is a tuple of M elements, each element is (nc, nd, amp)
        # Build the global pattern: interleave nc, nd
        pattern_list = []
        amp_product = 1.0
        for (nc, nd, amp) in combo:
            pattern_list.append(nc)
            pattern_list.append(nd)
            amp_product *= amp
        yield tuple(pattern_list), amp_product


def generate_input_states(M: int, Nc: int, N_max: int = None) -> List[Tuple[int, Tuple[int, ...]]]:
    """
    Return a list of all input states (k, r_tuple) for the given photon‑number sector.
    The list is in the canonical order used to index the density matrices.

    Parameters
    ----------
    M : int
        Number of mode pairs.
    Nc : int
        Total number of return photons in the sector.

    Returns
    -------
    states : list of (int, tuple of int)
        Each element is (k, (r0, r1, ..., r_{M-1})).
    """
    # Generate all return configurations r_tuple with sum = Nc
    # Using itertools.combinations_with_replacement for stars and bars
    # r_tuple is generated by iterating over all combinations of M-1 separators
    # among Nc + M - 1 positions.
    # A simpler way: use a recursive generator or use more_itertools.
    # We'll implement with a simple loop over all integer partitions.
    def gen_r_tuples(M, Nc, cap):
        if M == 1:
            if Nc <= (cap if cap is not None else Nc):
                yield (Nc,)
            return
        max_first = min(Nc, cap) if cap is not None else Nc
        for r0 in range(max_first + 1):
            for rest in gen_r_tuples(M - 1, Nc - r0, cap):
                yield (r0,) + rest

    states = []
    for k in range(M):
        for r_tup in gen_r_tuples(M, Nc, N_max):
            states.append((k, r_tup))
    return states

def chunkify(lst:list, n_chunks: int)->list[list[tuple[int, Any]]] :
    """Split lst into n_chunks roughly equal sublists, preserving index."""
    chunks = [[] for _ in range(n_chunks)]
    #This initializes your storage using a list comprehension. 
    # If n_chunks is 4, it creates a list containing 4 empty sublists: [[], [], [], []]. 
    # Each sublist will hold one chunk of wor
    for idx, item in enumerate(lst): # It loops through your original list. enumerate(lst) is the key here: 
        #instead of just giving you the item (the state), 
        # it also tracks its original position in the list, saving it in the variable idx (index)
        chunks[idx % n_chunks].append((idx, item))   # (global_index, (k, r_tuple))
    return chunks

def process_chunk(chunk: List[Tuple[int, Tuple[int, ...]]], M: int,  Nc: int,  local_BS: Dict[Tuple[int, int], List[Tuple[int, int, float]]]) -> List[Tuple[Tuple[int, ...], int, float]]:
    """
    Process a specific subset (chunk) of quantum input states.
    
    This function acts as a parallel worker. It iterates through a chunk of
    pre-assigned input states, calculates all reachable global output patterns
    and their amplitudes for each state, and tags the results with their original
    global indices for downstream re-sorting.

    Parameters
    ----------
    chunk : list of tuple
        A list where each element is a tuple: (global_idx, (k, r_tuple)).
        - global_idx (int): The original canonical index of this state.
        - k (int): The mode index containing the idler photon.
        - r_tuple (tuple of int): Photon configuration per mode pair.
    M : int
        Number of mode pairs in the system.
    Nc : int
        Total number of return photons in this sector.
    local_BS : dict
        Precomputed local beam‑splitter amplitude lookup table.
        Maps (nI, nR) to a list of (nc, nd, amplitude).

    Returns
    -------
    results : list of tuple
        A list of generated output data. Each element is a flat tuple:
        (pattern, global_idx, amplitude)
        - pattern (tuple of int): Flat global output configuration.
        - global_idx (int): The index of the input state that created it.
        - amplitude (float): The quantum amplitude of this specific combination.
    """
    results: List[Tuple[Tuple[int, ...], int, float]] = []
    
    for global_idx, (k, r_tuple) in chunk:
        # Generate every possible global output pattern for this specific state
        for pattern, amp in generate_patterns_for_input_state(k, r_tuple, local_BS, M):
            # Record the result, pinning the original global_idx to it
            results.append((pattern, global_idx, amp))
            
    return results

def build_pattern_amplitudes_parallel(M: int, Nc: int, local_BS: Dict[Tuple[int, int], List[Tuple[int, int, float]]], N_max: int,num_workers: int = 4) -> Dict[Tuple[int, ...], List[Tuple[int, float]]]:
    """
    Orchestrate the parallel generation and aggregation of quantum pattern amplitudes.

    This function generates all valid quantum input states, splits them into 
    balanced workloads (chunks), distributes those workloads across multiple 
    CPU cores, and finally aggregates all partial results into a global dictionary 
    grouped by output pattern.

    Parameters
    ----------
    M : int
        Number of mode pairs.
    Nc : int
        Total number of return photons in the sector.
    local_BS : dict
        Precomputed local beam-splitter amplitude table.
    N_max : int
        Maximum photon cutoff parameter used by the state generator.
    num_workers : int, default 4
        The number of parallel CPU processes (workers) to spawn.

    Returns
    -------
    pattern_amps : dict
        A dictionary where:
        - Key: Global output pattern (tuple of ints, e.g., (nc0, nd0, nc1, nd1...))
        - Value: A list of tuples, each containing (input_state_idx, amplitude)
                 showing which inputs contributed to that output pattern and by how much.
    """
    # Step 1: Generate all baseline states
    input_states: List[Tuple[int, Tuple[int, ...]]] = generate_input_states(M, Nc, N_max)

    # Step 2: Slice the workload into even piles for the workers
    chunks: List[List[Tuple[int, Tuple[int, Tuple[int, ...]]]]] = chunkify(input_states, num_workers)

    # Step 3: Package up the parameters for each process
    worker_args: List[Tuple[List[Any], int, int, Dict[Any, Any]]] = [(chunk, M, Nc, local_BS) for chunk in chunks]

    # Step 4: Fire up the parallel processing engine
    with Pool(num_workers) as pool:
        # starmap automatically unpacks each tuple in worker_args into process_chunk
        chunk_results: List[List[Tuple[Tuple[int, ...], int, float]]] = pool.starmap(process_chunk, worker_args)

    # Step 5: Gather and re-group the scattered data
    pattern_amps: defaultdict[Tuple[int, ...], List[Tuple[int, float]]] = defaultdict(list)
    
    for partial_list in chunk_results:
        for pattern, idx, amp in partial_list:
            # Group by the physical output pattern
            pattern_amps[pattern].append((idx, amp))

    return dict(pattern_amps)

def build_pattern_amplitudes_serial(M: int,Nc: int,local_BS: Dict[Tuple[int, int], List[Tuple[int, int, float]]],N_max: int = None) -> Dict[Tuple[int, ...], List[Tuple[int, float]]]:
    """
    Serial (single‑core) aggregation of output pattern amplitudes for all
    input states of a given Nc sector.

    Parameters
    ----------
    M : int
        Number of mode pairs.
    Nc : int
        Total number of return photons in the sector.
    local_BS : dict
        Precomputed local beam‑splitter amplitude table.
    N_max : int, optional
        Maximum photon number per mode (unused, kept for interface compatibility).

    Returns
    -------
    pattern_amps : dict
        Keys: global output pattern as a flat tuple of 2M integers.
        Values: list of (input_state_index, amplitude) for all input states
        that have a non‑zero amplitude for that pattern.
    """
    # 1. Generate the canonical list of input states (k, r_tuple)
    input_states = generate_input_states(M, Nc, N_max)   # this function must exist

    # 2. Prepare a default dictionary to collect (index, amplitude) per pattern
    pattern_amps = defaultdict(list)

    # 3. Loop over input states sequentially
    for idx, (k, r_tuple) in enumerate(input_states):
        for pattern, amp in generate_patterns_for_input_state(k, r_tuple, local_BS, M):
            pattern_amps[pattern].append((idx, amp))

    return dict(pattern_amps)

def compute_probabilities_for_pattern(entries: List[Tuple[int, float]],rho_abs: np.ndarray,rho_pres: np.ndarray, d_in: int) -> Tuple[float, float, float]:
    """
    Given the sparse amplitude list (idx, amp) for a single output pattern,
    build the dense vector A and compute:
        p0 = A† ρ_abs A   (target absent)
        p1 = A† ρ_pres A  (target present)
        Λ  = ln(p1/p0)

    Returns (Λ, p0, p1).
    """
    A = np.zeros(d_in, dtype=complex)
    for idx, amp in entries:
        A[idx] = amp

    p0 = np.real(np.dot(np.conj(A), np.dot(rho_abs, A)))
    p1 = np.real(np.dot(np.conj(A), np.dot(rho_pres, A)))

    if p0 <= 0 or p1 <= 0:
        lam = 0.0
    else:
        lam = np.log(p1 / p0)

    return lam, p0, p1




def process_one_block(args: Tuple) -> Tuple[int, Dict[Tuple[int, ...], Tuple[float, float, float]]]:
    """
    Worker function that processes one Nc block.
    Input args: (M, Nc, local_BS, N_max, rho_abs_block, rho_pres_block, inner_workers, use_slm, phases)
    Returns (Nc, lookup_table_for_this_block).
    """
    (M, Nc, local_BS, N_max, rho_abs_block, rho_pres_block,
     inner_workers, use_slm, phases) = args

    d_in = rho_abs_block.shape[0]

    # Optionally apply SLM phases
    if use_slm:
        states = generate_input_states(M, Nc, N_max)
        k_array = np.array([k for k, _ in states], dtype=int)
        rho_abs_block = apply_slm(rho_abs_block, k_array, phases)
        rho_pres_block = apply_slm(rho_pres_block, k_array, phases)

    # Build amplitude dictionary – always parallel if inner_workers > 1
    if inner_workers>1:
        pattern_amps = build_pattern_amplitudes_parallel( M, Nc, local_BS, N_max, num_workers=inner_workers)
    else:
        pattern_amps = build_pattern_amplitudes_serial( M, Nc, local_BS, N_max)


    # Compute probabilities and Λ for every output pattern
    block_table = {}
    for pattern, entries in pattern_amps.items():
        lam, p0, p1 = compute_probabilities_for_pattern(
            entries, rho_abs_block, rho_pres_block, d_in
        )
        block_table[pattern] = (lam, p0, p1)

    return Nc, block_table


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




def build_global_lookup_table(
    M: int,
    N_max: int,
    local_BS: dict,
    rho_pres_blocks: Dict[int, np.ndarray],
    rho_abs_blocks: Dict[int, np.ndarray],
    Nc_list: List[int] = None,
    outer_workers: int = 1,
    inner_workers: int = 1,
    use_slm: bool = False,
    phases: np.ndarray = None,
) -> Dict[int, Dict[Tuple[int, ...], Tuple[float, float, float]]]:
    """
    Build the complete log‑likelihood lookup table for all requested Nc sectors.

    Parameters
    ----------
    M, N_max : protocol size.
    local_BS : precomputed per‑pair BS amplitude table.
    rho_pres_blocks, rho_abs_blocks : density matrices for each Nc.
    Nc_list : list of Nc values to include (default: all keys in rho_pres_blocks).
    outer_workers : number of processes for block‑level parallelism.
    inner_workers : number of processes inside each block for amplitude aggregation.
    use_slm : whether to apply SLM phases before computing probabilities.
    phases : array of M phases for the SLM (if use_slm=True).

    Returns
    -------
    global_lookup : nested dict {Nc: {pattern: (lam, p0, p1)}}
    """
    if Nc_list is None:
        Nc_list = sorted(rho_pres_blocks.keys())

    tasks = []
    for Nc in Nc_list:
        tasks.append((
            M, Nc, local_BS, N_max,
            rho_abs_blocks[Nc], rho_pres_blocks[Nc],
            inner_workers, use_slm, phases
        ))

    if outer_workers > 1:
        with Pool(outer_workers) as pool:
            results = pool.map(process_one_block, tasks)
    else:
        results = [process_one_block(task) for task in tasks]

    global_lookup = {}
    for Nc, block_table in results:
        global_lookup[Nc] = block_table
    return global_lookup




def convert_lookup_to_json_serializable(lookup: Dict) -> Dict:
    """Convert lookup table to a JSON‑serializable dict.
    Keys: Nc -> str(pattern) -> [lam, p0, p1]"""

    json_lookup = {}
    for Nc, block in lookup.items():
        json_block = {}
        for pattern, (lam, p0, p1) in block.items():
            # Convert tuple to comma‑separated string
            key_str = ",".join(map(str, pattern))
            json_block[key_str] = [lam, p0, p1]
        json_lookup[str(Nc)] = json_block   # Nc key as string
    return json_lookup


async def download_file(remote_path: str, local_path: str):
    with open(local_path, "wb") as f:
        await volume.read_file_into_fileobj(remote_path, f)



if __name__ == "__main__":
    # --- Parameters for the simulation ---
    M = 6
    Nmax = 2
    kappa = 0.05      # target reflectivity (used in target‑present generation)
    Nbar = 0.5        # thermal background
    samples=1000

    VOLUME_NAME = "qi-results"
    volume = modal.Volume.from_name(VOLUME_NAME)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # remote paths (as before)
        pres_remote = f"M={M}_K={samples}_Nbar={Nbar}_Nmax={Nmax}/blocks.npz"
        abs_remote  = f"Target_Absent/M={M}_Nbar={Nbar}_Nmax={Nmax}/blocks.npz"
        bs_remote   = f"local_BS_tables/local_BS_table_Nmax={Nmax}/local_BS_Nmax{Nmax}.pkl"

        # local file paths
        pres_local = tmp / "pres.npz"
        abs_local  = tmp / "abs.npz"
        bs_local   = tmp / "local_BS.pkl"

        # Download using Modal CLI
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, pres_remote, str(pres_local)], check=True)
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, abs_remote,  str(abs_local)],  check=True)
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, bs_remote,   str(bs_local)],   check=True)

        pres_data = np.load(str(pres_local))
        rho_pres = {int(k.split('_')[1]): pres_data[k] for k in pres_data.files}
        total_trace=0.0 
        for Nc, block in rho_pres.items():
            total_trace= total_trace+ np.trace(block).real
        
        print(f"total trace of the block representation is : {total_trace}" )

        abs_data = np.load(str(abs_local))
        rho_abs = {int(k.split('_')[1]): abs_data[k] for k in abs_data.files}

        with open(str(bs_local), "rb") as f:
            local_BS = pickle.load(f)
        
        Nc_list = sorted(rho_pres.keys())
        start_time = time.time()
        global_lookup=  build_global_lookup_table(M=M,
            N_max=Nmax,
            local_BS=local_BS,
            rho_pres_blocks=rho_pres,
            rho_abs_blocks=rho_abs,
            Nc_list=Nc_list,
            outer_workers=1,   # adjust for your device/specs
            inner_workers=10,
            use_slm=False,
        )

        end_time= time.time()
        print(f"Time for M={M} and Nmax= {Nmax} is: {end_time - start_time} seconds ")


        pickle_path = tmp / "lookup.pkl"

        with open(str(pickle_path), "wb") as f:
            pickle.dump(global_lookup, f)
        
        json_serializable = convert_lookup_to_json_serializable(global_lookup)
        json_path = tmp / "lookup.json"

        with open(str(json_path), "w") as f:
            json.dump(json_serializable, f, indent=2)
        
        remote_dir = f"Global_lookuptable/M={M}_Nmax={Nmax}"
        print(f"Uploading lookup table to volume at {remote_dir}/ ...")
        with volume.batch_upload() as batch:
            batch.put_file(str(pickle_path), f"{remote_dir}/lookup.pkl")
            batch.put_file(str(json_path), f"{remote_dir}/lookup.json")


    