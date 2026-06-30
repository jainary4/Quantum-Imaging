from src.POVM.POVM_interferometric import build_local_BS_table
import modal
import tempfile
from pathlib import Path
import json 
import pickle

VOLUME_NAME = "qi-results"          # your existing volume
N_MAX_VALUES = [8,9,10,11,12,13,14,15]     # the N_max values you want now
REMOTE_PARENT_DIR = "local_BS_tables"

def main():
    volume = modal.Volume.from_name(VOLUME_NAME)  # connect to the existing volume

    with tempfile.TemporaryDirectory() as tmpdir:
        upload_list = []   # (local_path, remote_path) pairs

        for Nmax in N_MAX_VALUES:
            print(f"Building table for Nmax = {Nmax} ...")
            table = build_local_BS_table(Nmax)

            # Create the directory for this Nmax: local_BS_table_Nmax={Nmax}
            dir_name = f"local_BS_table_Nmax={Nmax}"
            local_dir = Path(tmpdir) / dir_name
            local_dir.mkdir(parents=True, exist_ok=True)

            # Save as pickle
            pkl_filename = f"local_BS_Nmax{Nmax}.pkl"
            pkl_path = local_dir / pkl_filename
            with open(pkl_path, "wb") as f:
                pickle.dump(table, f)

            # Save as JSON (keys are converted to strings)
            json_filename = f"local_BS_Nmax{Nmax}.json"
            json_path = local_dir / json_filename
            with open(json_path, "w") as f:
                json.dump({str(k): v for k, v in table.items()}, f, indent=2)

            # Record the remote path (preserving the directory structure)
            remote_dir = f"{REMOTE_PARENT_DIR}/{dir_name}"
            upload_list.append((str(pkl_path), f"{remote_dir}/{pkl_filename}"))
            upload_list.append((str(json_path), f"{remote_dir}/{json_filename}"))

        # ------------------------------------------------------------------
        # 2. Upload everything to the Modal volume in one batch
        # ------------------------------------------------------------------
        print("Uploading files to Modal volume ...")
        with volume.batch_upload() as batch:
            for local_path, remote_path in upload_list:
                batch.put_file(local_path, remote_path)
        print("Upload complete.")


if __name__ == "__main__":
    main()