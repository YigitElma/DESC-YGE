#!/usr/bin/env python3
"""
Read both result folders, plot PR vs master into compare.png,
write a Markdown summary to commit_msg.txt,
and embed the plot in the *job summary* (GITHUB_STEP_SUMMARY).
"""
import base64, os, numpy as np, matplotlib.pyplot as plt
import pickle

print(os.getcwd())
print("Files in the current directory:")
for file in os.listdir("."):
    print(file)

with open("master.pickle", "rb") as f:
    data_master = pickle.load(f)
with open("pr.pickle", "rb") as f:
    data_pr = pickle.load(f)

# ---------- plot ----------
plt.figure(figsize=(12, 6))
for i, name in enumerate(data_master.keys()):
    plt.plot(data_pr[name]["t"], data_pr[name]["mem"], "r", label=f"{name} PR", lw=3)
    plt.plot(
        data_master[name]["t"],
        data_master[name]["mem"],
        "--b",
        label=f"{name} Master",
        lw=1,
    )
plt.xlabel("Time [s]")
plt.ylabel("Δ RSS [MB]")
plt.title("Memory comparison (PR vs master)")
plt.grid(True)
plt.legend()
plt.tight_layout()
PNG = "compare.png"
plt.savefig(PNG, dpi=100)

# ---------- commit message ----------
msg = f"### Memory benchmark result\n\n```diff\n"
msg += (
    f"| {'Test Name':^22} | {'Master':^22} | {'PR':^22} | {'Δ (MB)':^22} | {'%Δ':^22} |"
)
msg += f"\n| {'-'*22} | {'-'*22} | {'-'*22} | {'-'*22} | {'-'*22} |\n"
for i, name in enumerate(data_master.keys()):
    peak_pr = data_pr[name]["mem"].max()
    peak_ma = data_master[name]["mem"].max()
    delta = peak_pr - peak_ma
    percent_change = (delta / peak_ma) * 100
    sign = "-" if delta >= 0 else "+"
    msg += (
        f"{sign} {name:^22} | {peak_ma:.1f} | {peak_pr:.1f} |"
        + f" {sign}{abs(delta):.1f} | {sign}{abs(percent_change):.2f}% |\n"
    )
msg += f"```"

with open("commit_msg.txt", "w") as fh:
    fh.write(msg)

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
