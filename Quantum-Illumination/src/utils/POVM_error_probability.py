import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import tempfile
import modal

def download_lookup_json_from_volume(volume_name: str,remote_path: str,local_dir: Optional[Path] = None) -> Path:
    """
    Download a file from a Modal volume using the Modal CLI.

    Parameters
    ----------
    volume_name : str
        Name of the Modal volume (e.g., "qi-results").
    remote_path : str
        Path inside the volume, e.g., "Global_lookuptable/M=2_Nmax=2/lookup.json".
    local_dir : Path, optional
        Directory in which to store the downloaded file.
        Defaults to a temporary directory.

    Returns
    -------
    local_file : Path
        Path to the downloaded file on the local disk.
    """
    if local_dir is None:
        local_dir = Path(tempfile.mkdtemp())

    local_file = local_dir / Path(remote_path).name

    subprocess.run(
        ["modal", "volume", "get", volume_name, remote_path, str(local_file)],
        check=True,
        text=True,
    )
    return local_file



def load_lookup_table_from_json(json_path: Path) -> Dict[int, Dict[Tuple[int, ...], Tuple[float, float, float]]]:
    """
    Load a JSON‑serialised lookup table and convert string keys to native types.

    The JSON structure is expected to be:
    { "Nc": { "n0,n1,n2,...": [lam, p0, p1], ... }, ... }

    Parameters
    ----------
    json_path : Path
        Path to the JSON file.

    Returns
    -------
    lookup : dict
        Nested dictionary: lookup[Nc][pattern_tuple] = (lam, p0, p1)
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    lookup: Dict[int, Dict[Tuple[int, ...], Tuple[float, float, float]]] = {}

    for nc_str, block in raw.items():
        Nc = int(nc_str)
        lookup[Nc] = {}
        for pat_str, (lam, p0, p1) in block.items():
            # "0,1,2,3" -> (0,1,2,3)
            pattern = tuple(int(x) for x in pat_str.split(","))
            lookup[Nc][pattern] = (lam, p0, p1)

    return lookup


def compute_error_probability(lookup: Dict[int, Dict[Tuple[int, ...], Tuple[float, float, float]]],threshold: float = 0.0) -> Dict[str, float]:
    """
    Compute the exact error probability from the lookup table (no sampling).

    Parameters
    ----------
    lookup : dict
        The global lookup table.
    threshold : float
        Decision threshold for the log‑likelihood ratio.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - "P_FA" : false‑alarm probability
        - "P_MD" : missed‑detection probability
        - "P_e"  : total error probability (equal priors)
    """
    pfa = 0.0
    pmd = 0.0

    for block in lookup.values():
        for lam, p0, p1 in block.values():
            if lam > threshold:
                pfa += p0      # declare present when absent
            else:
                pmd += p1      # declare absent when present

    pe = 0.5 * pfa + 0.5 * pmd
    return {"P_FA": pfa, "P_MD": pmd, "P_e": pe}


if __name__ == "__main__":
    # --- Parameters for the lookup table we want to load ---
    M = 6
    Nmax = 2
    VOLUME_NAME = "qi-results"

    # The remote path as stored by your build script
    remote_json = f"Global_lookuptable/M={M}_Nmax={Nmax}/lookup_exact.json"

    print(f"Downloading lookup table from Modal volume '{VOLUME_NAME}'...")
    json_file = download_lookup_json_from_volume(VOLUME_NAME, remote_json)

    print("Loading and converting table...")
    lookup = load_lookup_table_from_json(json_file)

    # --- Compute exact error probability ---
    exact = compute_error_probability(lookup)
    print("\n--- Exact analytical error ---")
    print(f"  P_FA = {exact['P_FA']:.6f}")
    print(f"  P_MD = {exact['P_MD']:.6f}")
    print(f"  P_e  = {exact['P_e']:.6f}")
