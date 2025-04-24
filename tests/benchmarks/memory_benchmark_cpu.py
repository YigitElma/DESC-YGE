#!/usr/bin/env python3

import subprocess
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import psutil
import gc
import sys


def monitor_ram(proc, interval, ram_usage, timestamps):
    """Sample system RAM until *proc* finishes."""
    while proc.poll() is None:  # child still running?
        info = psutil.virtual_memory()
        used_mb = (info.total - info.available) / 1024 / 1024
        ram_usage.append(used_mb)
        timestamps.append(time.time())
        time.sleep(interval)

    # keep watching for an extra second
    end = time.time() + 1.0
    while time.time() < end:
        info = psutil.virtual_memory()
        ram_usage.append((info.total - info.available) / 1024 / 1024)
        timestamps.append(time.time())
        time.sleep(interval)


def monitor_vram(proc, interval, vram_usage, timestamps):
    """Sample total GPU memory until *proc* finishes."""
    while proc.poll() is None:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        vram_usage.append(int(out.split()[0]))
        timestamps.append(time.time())
        time.sleep(interval)

    # keep watching for an extra second
    end = time.time() + 2.0
    while time.time() < end:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        vram_usage.append(int(out.split()[0]))
        timestamps.append(time.time())
        time.sleep(interval)


if __name__ == "__main__":
    mode = "CPU"  # "CPU" or "GPU"
    interval = 0.1  # seconds between samples
    res = 7

    mems = []
    ts = []

    funs = [
        "proximal_freeb_compute",
        "proximal_freeb_jac",
    ]

    for i in range(len(funs)):
        mem = []
        t = []
        gc.collect()
        # start the sampler thread
        # launch the script to be profiled
        child = subprocess.Popen(["python", "memory_funcs.py", funs[i], f"{res}", mode])
        target = monitor_vram if mode == "GPU" else monitor_ram
        sampler = threading.Thread(
            target=target,
            args=(child, interval, mem, t),
            daemon=True,
        )
        sampler.start()

        # wait until the child exits, then join the sampler
        child.wait()
        sampler.join()
        mems.append(mem - min(mem))
        ts.append(t - t[0])

    branch = sys.argv[1]  # master or pr
    # plotting
    mem_usage = np.asarray(mems)
    times = np.asarray(ts)
    np.savetxt(f"{branch}_memory.txt", mem_usage)
    np.savetxt(f"{branch}_time.txt", times)

    plt.figure(figsize=(15, 7))
    plt.plot(times, mem_usage, label=mode, color="orange")
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Memory Usage (MB)", fontsize=20)
    plt.title(f"Memory usage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"memory.png", dpi=300)
    plt.show()
