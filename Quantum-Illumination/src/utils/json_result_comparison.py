import modal
import json
import csv
import sys
import os

app = modal.App("compare-lookups")
volume = modal.Volume.from_name("qi-results")

# Remote paths to your two JSON lookup tables (adjust if needed)
BLOCK_REMOTE_PATH = "Global_lookuptable/M=6_Nmax=2/lookup_exact.json"
SIGMA_REMOTE_PATH = "Global_lookuptable/M=6_Nmax=2/lookup_sigma_exact.json"
OUTPUT_REMOTE_DIR = "comparisons"                     # will be created inside the volume
OUTPUT_FILENAME = "comparison_M=6_Nmax=2.csv"

@app.function(volumes={"/vol": volume})
def compare_and_save():
    """Read both lookup tables from the volume, compare, write CSV back to volume."""
    # 1. Load JSON files
    with open(f"/vol/{BLOCK_REMOTE_PATH}", "r") as f:
        block_data = json.load(f) 
    with open(f"/vol/{SIGMA_REMOTE_PATH}", "r") as f:
        sigma_data = json.load(f)

    # 2. Prepare output directory
    out_dir = f"/vol/{OUTPUT_REMOTE_DIR}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUTPUT_FILENAME)

    # 3. Comparison logic (same as before)
    all_nc = sorted(block_data.keys(), key=int)
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Nc", "pattern",
            "lam_block", "lam_sigma", "lam_ratio", "lam_diff",
            "p0_block", "p0_sigma", "p0_ratio", "p0_diff",
            "p1_block", "p1_sigma", "p1_ratio", "p1_diff"
        ])
        for nc_str in all_nc:
            block_block = block_data.get(nc_str, {})
            sigma_block = sigma_data.get(nc_str, {})
            all_patterns = sorted(set(block_block.keys()) | set(sigma_block.keys()))
            for pat in all_patterns:
                lam_b, p0_b, p1_b = block_block.get(pat, [0.0, 0.0, 0.0])
                lam_s, p0_s, p1_s = sigma_block.get(pat, [0.0, 0.0, 0.0])

                lam_ratio = lam_s / lam_b if abs(lam_b) > 1e-15 else float('inf')
                p0_ratio = p0_s / p0_b if p0_b > 1e-15 else float('inf')
                p1_ratio = p1_s / p1_b if p1_b > 1e-15 else float('inf')

                writer.writerow([
                    nc_str, pat,
                    f"{lam_b:.10f}", f"{lam_s:.10f}", f"{lam_ratio:.10f}", f"{lam_s - lam_b:.2e}",
                    f"{p0_b:.10f}", f"{p0_s:.10f}", f"{p0_ratio:.10f}", f"{p0_s - p0_b:.2e}",
                    f"{p1_b:.10f}", f"{p1_s:.10f}", f"{p1_ratio:.10f}", f"{p1_s - p1_b:.2e}",
                ])
    return f"CSV saved to volume at /{OUTPUT_REMOTE_DIR}/{OUTPUT_FILENAME}"

@app.local_entrypoint()
def main():
    # 1. Run the comparison on Modal
    msg = compare_and_save.remote()
    print(msg)

    # 2. Download result using Modal Volume SDK (no subprocess)
    local_file = OUTPUT_FILENAME
    remote_file = f"{OUTPUT_REMOTE_DIR}/{OUTPUT_FILENAME}"
    try:
        # Read the file content directly from the volume
        data = volume.read_file(remote_file)
        with open(local_file, "wb") as f:
            f.write(data)
        print(f"Downloaded {local_file} to your local machine.")
    except Exception as e:
        print(f"Failed to download file: {e}")