from POVM_error_probability import download_lookup_json_from_volume , load_lookup_table_from_json, compute_error_probability
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import tempfile
import modal

M= [2,3,4,5,6]
Nmax=2
VOLUME_NAME = "qi-results"
P_FA=[]
P_MD=[]
P_e=[]

for m in M:

    remote_json = f"Global_lookuptable/M={m}_Nmax={Nmax}/lookup.json"

    print(f"Downloading lookup table from Modal volume '{VOLUME_NAME}'...")
    json_file = download_lookup_json_from_volume(VOLUME_NAME, remote_json)

    print("Loading and converting table...")
    lookup = load_lookup_table_from_json(json_file)

    # --- Compute exact error probability ---
    exact = compute_error_probability(lookup)
    P_FA.append(exact['P_FA'])
    P_MD.append(exact['P_MD'])
    P_e.append(exact['P_e'])
    print(f"P_FA = {exact['P_FA']:.6f}")
    print(f"P_MD = {exact['P_MD']:.6f}")
    print(f"P_e = {exact['P_e']:.6f}")


"""fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(16, 6))

ax1.plot(M, P_FA, marker='o', color='green', linewidth=2)
ax1.set_title(f'M vs Probability of False Alarm graph ')
ax1.set_xlabel('Number of Modes ($M$)')
ax1.set_ylabel('Probability of False Alarm') 
ax1.set_xticks(M)
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()


ax2.plot(M, P_MD, marker='o', color='green', linewidth=2)
ax2.set_title(f'M vs Probability of Missed Detection Alarm graph ')
ax2.set_xlabel('Number of Modes ($M$)')
ax2.set_ylabel('Probability of Missed Detection') 
ax2.set_xticks(M)
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

ax3.plot(M, P_e, marker='o', color='green', linewidth=2)
ax3.set_title(f'M vs Probability of error graph ')
ax3.set_xlabel('Number of Modes ($M$)')
ax3.set_ylabel('Probability of error= 0.5 (P_MD) + 0.5 (P_FA)') 
ax3.set_xticks(M)
ax3.grid(True, which="both", ls="--", alpha=0.5)
ax3.legend()

plt.tight_layout()
plt.savefig('scaling_analysis_POVM.png', dpi=300)
plt.show()"""

