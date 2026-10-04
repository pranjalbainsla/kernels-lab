"""Builds every reduction kernel for n = 2^log2n elements, runs it, and prints a markdown table.

Usage:
    python reduce_report.py --n 28     # compile all kernels with n = 1 << 28, run, print table
    python reduce_report.py            # defaults to --n 22
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL_DIR = Path(__file__).parent / "kernels_impl" / "reduction"

# source file stem -> label shown in the table
KERNELS = {
    "reduce1": "Kernel 1: interleaved (divergent)",
    "reduce2": "Kernel 2: interleaved (bank conflicts)",
    "reduce3": "Kernel 3: sequential addressing",
    "reduce4": "Kernel 4: first add during load",
    "reduce5": "Kernel 5: unroll last warp",
    "reduce6": "Kernel 6: completely unrolled",
    "reduce7": "Kernel 7: multiple elements/thread",
}


def build_and_run(stem, log2n, build_dir, nvcc_args):
    exe = build_dir / stem
    cmd = ["nvcc", "-O3", f"-DLOG2N={log2n}", *nvcc_args, str(KERNEL_DIR / f"{stem}.cu"), "-o", str(exe)]
    subprocess.run(cmd, check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    if "Verification successful" not in out:
        print(f"warning: {stem} failed verification:\n{out}", file=sys.stderr)
    t = float(re.search(r"Avg time:\s*([\d.]+)", out).group(1))
    bw = float(re.search(r"Effective bandwidth:\s*([\d.]+)", out).group(1))
    return t, bw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=22, help="log2 of the element count (n = 1 << n)")
    ap.add_argument("--nvcc-args", nargs=argparse.REMAINDER, default=[],
                    help="extra nvcc flags, e.g. --nvcc-args -arch=sm_75 (must come last)")
    args = ap.parse_args()

    data = {}
    with tempfile.TemporaryDirectory() as tmp:
        for stem, name in KERNELS.items():
            print(f"building and running {stem} (n = 1 << {args.n})...", file=sys.stderr)
            data[name] = build_and_run(stem, args.n, Path(tmp), args.nvcc_args)

    times = [t for t, _ in data.values()]
    header = ("Kernel", "Time (ms)", "Bandwidth (GB/s)", "Step speedup", "Cumulative")
    lines = ["| " + " | ".join(header) + " |", "|---|:---:|:---:|:---:|:---:|"]
    for i, (name, (t, bw)) in enumerate(data.items()):
        step = "" if i == 0 else f"{times[i - 1] / t:.2f}x"
        cum = "" if i == 0 else f"{times[0] / t:.2f}x"
        lines.append(f"| {name} | {t:.4f} | {bw:.1f} | {step} | {cum} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
