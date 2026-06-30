import numpy as np
from typing import Dict, List, Tuple
import modal
import tempfile 
from pathlib import Path


VOLUME_NAME = "qi-results"




def generate_input_states(M: int, Nc: int, N_max: int = None) -> List[Tuple[int, Tuple[int, ...]]]:
    """
    Return all input basis states (k, r_tuple) for a given photon‑number sector.

    Parameters
    ----------
    M : int
        Number of mode pairs.
    Nc : int
        Total number of return photons in the sector.
    N_max : int, optional
        Maximum allowed photons per mode (unused here as Nc <= M * N_max).

    Returns
    -------
    states : list of (int, tuple of int)
        Each element is (k, (r0, r1, ..., r_{M-1})).
        The list is in canonical order: increasing k, then lexicographic r_tuple.
    """
    def gen_r_tuples(M: int, Nc: int, cap: int) -> List[Tuple[int, ...]]:

        if M == 1:
            if Nc <= (cap if cap is not None else Nc):
                yield (Nc,)
            return
        max_first = min(Nc, cap) if cap is not None else Nc
        for r0 in range(max_first + 1):
            for rest in gen_r_tuples(M - 1, Nc - r0, cap):
                yield (r0,) + rest

    states: List[Tuple[int, Tuple[int, ...]]] = []
    for k in range(M):
        for r_tuple in gen_r_tuples(M, Nc, N_max):
            states.append((k, r_tuple))
    return states


# ----------------------------------------------------------------------
# Target‑absent block generation
# ----------------------------------------------------------------------

def thermal_pmf(n: int, Nbar: float, Nmax: int) -> float:
    """
    Bose‑Einstein (geometric) probability for a single thermal mode,
    **not truncated**, i.e. the ideal infinite‑temperature distribution.

    Parameters
    ----------
    n : int
        Photon number.
    Nbar : float
        Mean photon number of the thermal state.
    Nmax : int
        Fock truncation (unused, kept for signature compatibility).

    Returns
    -------
    float
        P(n) = (Nbar^n) / (Nbar+1)^(n+1).
    """
    return (Nbar ** n) / ((Nbar + 1) ** (n + 1))


def generate_target_absent_blocks(
    M: int,
    Nmax: int,
    Nbar: float
) -> Dict[int, np.ndarray]:
    """
    Build the block‑diagonal target‑absent density matrices.

    Each Nc sector's matrix is diagonal in the canonical input basis
    and its entries are

        P(k, r_tuple) = (1 / M) * ∏_{j=0}^{M-1} P_thermal(r_j)

    where P_thermal is the truncated, normalised thermal distribution.

    Parameters
    ----------
    M : int
        Number of mode pairs.
    Nmax : int
        Maximum photon number per mode (Fock truncation).
    Nbar : float
        Mean photon number of the thermal background.

    Returns
    -------
    blocks : dict
        Keys are total return photon numbers Nc (int).
        Values are complex ndarrays of shape (d_in, d_in), where
        d_in = M * C(Nc+M-1, Nc).  Each matrix is purely diagonal
        (no off‑diagonal coherences).
    """
    d: int = Nmax + 1
    blocks: Dict[int, np.ndarray] = {}

    # Pre‑compute the normalised thermal distribution for one mode
    n_values: np.ndarray = np.arange(d, dtype=int)
    raw_probs: np.ndarray = np.array([thermal_pmf(n, Nbar, Nmax) for n in n_values])
    thermal_probs: np.ndarray = raw_probs / raw_probs.sum()  # normalised

    max_Nc: int = M * Nmax

    for Nc in range(max_Nc + 1):
        # Get all basis states (k, r_tuple) for this sector
        states: List[Tuple[int, Tuple[int, ...]]] = generate_input_states(M, Nc, Nmax)
        d_in: int = len(states)
        if d_in == 0:
            continue

        # Diagonal density matrix for the block
        rho_block: np.ndarray = np.zeros((d_in, d_in), dtype=complex)

        for idx, (k, r_tuple) in enumerate(states):
            # Probability = (1/M) * product of thermal probabilities
            prob: float = 1.0 / M
            for r_j in r_tuple:
                prob *= thermal_probs[r_j]
            rho_block[idx, idx] = prob

        blocks[Nc] = rho_block

    return blocks


# ----------------------------------------------------------------------
# Example usage (if run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    M = 9
    Nmax = 2
    Nbar = 0.5

    # Generate the target‑absent blocks
    rho_abs_blocks = generate_target_absent_blocks(M, Nmax, Nbar)

    print("Block keys (Nc):", list(rho_abs_blocks.keys()))
    for Nc, block in rho_abs_blocks.items():
        print(f"Nc = {Nc}, block = {block}, trace = {np.trace(block).real:.6f}")
    
    volume = modal.Volume.from_name(VOLUME_NAME)
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_name = f"M={M}_Nbar={Nbar}_Nmax={Nmax}"
        local_dir = Path(tmpdir) / dir_name
        local_dir.mkdir(parents=True, exist_ok=True)
        npz_path= local_dir/"blocks.npz"
        np.savez_compressed(npz_path, **{f"Nc_{Nc}": block for Nc, block in rho_abs_blocks.items()})
        remote_dir = f"Target_Absent/{dir_name}"
        remote_path = f"{remote_dir}/blocks.npz"

        # Upload
        with volume.batch_upload() as batch:
            batch.put_file(str(npz_path), remote_path)

        print(f"Uploaded to volume '{VOLUME_NAME}' at '{remote_path}'")



