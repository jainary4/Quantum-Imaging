import numpy as np
import modal
from typing import Dict, List, Tuple
import modal
import tempfile 
from pathlib import Path

VOLUME_NAME = "qi-results"

def thermal_distribution(Nbar: float, Nmax: int) -> np.ndarray:
    """Normalised thermal photon-number distribution truncated to [0, Nmax]."""
    d = Nmax + 1
    n_vals = np.arange(d)
    raw = np.array([(Nbar**n) / ((Nbar + 1)**(n + 1)) for n in n_vals])
    return raw / raw.sum()

def generate_target_absent_sigmas(M: int, Nmax: int, Nbar: float) -> np.ndarray:
    """
    Return the averaged sigma matrices for the target-absent hypothesis.

    Parameters
    ----------
    M : int
        Number of mode pairs.
    Nmax : int
        Fock truncation (max photons per mode).
    Nbar : float
        Mean thermal photon number.

    Returns
    -------
    sigma_abs : np.ndarray of shape (M, 2, 2, d, d), dtype complex
        sigma_abs[m, q', q] is the d×d matrix for mode m.
        Only (0,0) and (1,1) are non-zero, both equal to diag(P_0,…,P_{Nmax}).
        (0,1) and (1,0) are zero matrices.
    """
    d = Nmax + 1
    probs = thermal_distribution(Nbar, Nmax)          # shape (d,)
    diag_thermal = np.diag(probs)                    # (d, d)

    sigma_abs = np.zeros((M, 2, 2, d, d), dtype=complex)
    sigma_abs[:, 0, 0] = diag_thermal               # σ^{(0,0)} for every mode
    sigma_abs[:, 1, 1] = diag_thermal               # σ^{(1,1)}
    # (0,1) and (1,0) stay zero

    return sigma_abs

if __name__ == "__main__":
    M = 15
    Nmax = 2
    Nbar = 0.5

    rho_abs_sigmas= generate_target_absent_sigmas(M,Nmax,Nbar)

    volume = modal.Volume.from_name(VOLUME_NAME)
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_name = f"M={M}_Nbar={Nbar}_Nmax={Nmax}"
        local_dir = Path(tmpdir) / dir_name
        local_dir.mkdir(parents=True, exist_ok=True)
        npy_path= local_dir/"sigmas.npy"
        np.save(npy_path, rho_abs_sigmas)
        remote_dir = f"Target_Absent/{dir_name}"
        remote_path = f"{remote_dir}/sigmas.npy"

        with volume.batch_upload() as batch:
            batch.put_file(str(npy_path), remote_path)