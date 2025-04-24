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
num_tests = len(data_master.keys())
fig, axes = plt.subplots(num_tests, 1, figsize=(12, 6 * num_tests), sharex=False)

for i, (name, ax) in enumerate(zip(data_master.keys(), axes)):
    ax.plot(data_pr[name]["t"], data_pr[name]["mem"], "r", label="PR", lw=3)
    ax.plot(
        data_master[name]["t"],
        data_master[name]["mem"],
        "--b",
        label="Master",
        lw=1,
    )
    ax.set_title(name)
    ax.set_ylabel("Δ RSS [MB]")
    max_time = max(data_master[name]["t"][-1], data_pr[name]["t"][-1]) + 0.5
    ax.set_xlabel(f"Time [s]")
    ax.set_xlim([0, max_time])
    ax.grid(True)
    ax.legend()
plt.tight_layout()
PNG = "compare.png"
plt.savefig(PNG, dpi=100)

# ---------- commit message ----------
msg = f"### Memory benchmark result\n\n```diff\n"
msg += f"| {'Test Name':^22} | {'Master (MB)':^18} | {'PR (MB)':^18} | {'Δ (MB)':^18} | {'%Δ':^18} |\n"
msg += f"| {'-'*22} | {'-'*18} | {'-'*18} | {'-'*18} | {'-'*18} |\n"
for i, name in enumerate(data_master.keys()):
    peak_pr = data_pr[name]["mem"].max()
    peak_ma = data_master[name]["mem"].max()
    delta = peak_pr - peak_ma
    percent_change = (delta / peak_ma) * 100
    sign = "-" if delta >= 0 else "+"
    msg += (
        f"{sign} {name:>22} | {peak_ma:^18.1f} | {peak_pr:^18.1f} |"
        + f" {sign}{abs(delta):^18.1f} | {f'{sign}{abs(percent_change)}%':^18.2f} |\n"
    )
msg += f"```"

with open("commit_msg.txt", "w") as fh:
    fh.write(msg)

# ---------- add the image to the job summary ----------
summary = os.getenv("GITHUB_STEP_SUMMARY")
if summary:
    with open(summary, "a") as out:
        out.write("\n## Memory timeline\n\n")
        out.write(f"![Memory plot]({PNG})\n")
