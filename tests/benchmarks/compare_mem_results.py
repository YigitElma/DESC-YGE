#!/usr/bin/env python3
"""
Read both result folders, plot PR vs master into compare.png,
write a Markdown summary to commit_msg.txt,
and embed the plot in the *job summary* (GITHUB_STEP_SUMMARY).
"""
import base64, os, numpy as np, matplotlib.pyplot as plt

mpr = np.loadtxt("pr_memory.txt.txt")
tpr = np.loadtxt("pr_time.txt")
mma = np.loadtxt("master_memory.txt.txt")
tma = np.loadtxt("master_time.txt")

# ---------- plot ----------
plt.figure(figsize=(12, 6))
plt.plot(tpr, mpr, label="PR", lw=1.5)
plt.plot(tma, mma, label="master", lw=1.5)
plt.xlabel("Time [s]")
plt.ylabel("Δ RSS [MB]")
plt.title("Memory comparison (PR vs master)")
plt.grid(True)
plt.legend()
plt.tight_layout()
PNG = "compare.png"
plt.savefig(PNG, dpi=150)

# ---------- numbers ----------
peak_pr = float(mpr.max())
peak_ma = float(mma.max())
delta = peak_pr - peak_ma
sign = "+" if delta >= 0 else "-"

# ---------- commit message ----------
with open("commit_msg.txt", "w") as fh:
    fh.write(
        f"""### Memory benchmark result

| metric          | master | PR | Δ |
|-----------------|:------:|:--:|---:|
| **peak RSS [MB]** | {peak_ma:.1f} | **{peak_pr:.1f}** | {sign}{abs(delta):.1f} |

A positive Δ means the pull request uses more memory.

_Detailed timeline is available in the “Memory Benchmarks / benchmark” check summary._
"""
    )

# ---------- add the image to the job summary ----------
summary = os.getenv("GITHUB_STEP_SUMMARY")
if summary:
    with open(PNG, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    with open(summary, "a") as out:
        out.write("\n## Memory timeline\n\n")
        out.write(
            f'<img src="data:image/png;base64,{b64}" alt="memory plot" width="800"/>\n'
        )
