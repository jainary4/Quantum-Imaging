import modal

app = modal.App("delete-files")
volume = modal.Volume.from_name("qi-results")

@app.function(volumes={"/vol": volume})
def delete():
    import os
    files = [
       "/vol/Global_lookuptable/M=6_Nmax=2/lookup.json",
       "/vol/Global_lookuptable/M=5_Nmax=2/lookup.pkl",
    ]
    for f in files:
        try:
            os.remove(f)
            print(f"Deleted {f}")
        except FileNotFoundError:
            print(f"File {f} not found, skipping.")

@app.local_entrypoint()
def main():
    delete.remote()