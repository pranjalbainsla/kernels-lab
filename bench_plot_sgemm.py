import re
import statistics
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline

KERNELS = [
    ("1 naive",         "kernel_impl/sgemm_naive.cu",       "launch_sgemm_naive"),
    ("2 coalesced",     "kernel_impl/sgemm_coalesced.cu",   "launch_sgemm_coalesced"),
    ("3 smem",          "kernel_impl/sgemm_smem.cu",        "launch_sgemm_smem"),
    ("4 1D blocktile",  "kernel_impl/sgemm_1D_blocktile.cu", "launch_sgemm_1D_blocktile"),
    ("5 2D blocktile",  "kernel_impl/sgemm_2D_blocktile.cu", "launch_sgemm_2D_blocktile"),
]
SIZES = [256, 512, 1024, 2048, 3072, 4096]   # all multiples of 64
WARMUP, ITERS = 3, 10

torch.backends.cuda.matmul.allow_tf32 = False   # FP32 cuBLAS baseline, TF32 off
torch.backends.cudnn.allow_tf32 = False
gpu = torch.cuda.get_device_name(0)
gpu_tag = re.sub(r"[^A-Za-z0-9]+", "_", gpu).strip("_")


def load_kernel(label, path, fn):
    src = open(path).read()
    decl = f"void {fn}(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta);"
    name = "ext_" + re.sub(r"\W+", "_", fn)
    return getattr(load_inline(name=name, cpp_sources=decl, cuda_sources=src,
                               functions=[fn], verbose=False), fn)


def time_ms(fn):
    """Median time in ms over ITERS runs, using CUDA events."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


funcs = {label: load_kernel(label, path, fn) for label, path, fn in KERNELS}
rows = []
for N in SIZES:
    torch.manual_seed(0)
    A = torch.randn(N, N, device="cuda")
    B = torch.randn(N, N, device="cuda")
    C = torch.zeros(N, N, device="cuda")
    ref = A @ B
    flops = 2 * N ** 3

    rows.append(("cuBLAS", N, flops / (time_ms(lambda: torch.matmul(A, B)) * 1e6)))
    for label, f in funcs.items():
        C.zero_()
        f(A, B, C, 1.0, 0.0)
        torch.cuda.synchronize()
        ok = torch.allclose(C, ref, rtol=1e-3, atol=1e-2)
        if not ok:
            print(f"WARNING: {label} wrong at N={N}, max err {(C - ref).abs().max().item():.3e}")
        ms = time_ms(lambda: f(A, B, C, 1.0, 0.0))
        rows.append((label, N, flops / (ms * 1e6)))   # GFLOPS = FLOPs / (ms * 1e-3) / 1e9
        print(f"N={N:5d}  {label:16s} {rows[-1][2]:9.1f} GFLOPS  {'ok' if ok else 'WRONG'}")

df = pd.DataFrame(rows, columns=["kernel", "N", "gflops"])
df.to_csv(f"results_{gpu_tag}.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
for label in [k[0] for k in KERNELS] + ["cuBLAS"]:
    d = df[df.kernel == label].sort_values("N")
    ax.plot(d.N, d.gflops, marker="o", ls="--" if label == "cuBLAS" else "-",
            color="k" if label == "cuBLAS" else None, label=label)
ax.set_xlabel("Matrix size N (M = N = K)")
ax.set_ylabel("GFLOPS/s (FP32, TF32 off)")
ax.set_title(f"SGEMM kernel ladder vs matrix size, {gpu}")
ax.set_xscale("log", base=2)
ax.set_xticks(SIZES)
ax.set_xticklabels(SIZES)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"sgemm_vs_size_{gpu_tag}.png", dpi=150)
print(f"wrote results_{gpu_tag}.csv and sgemm_vs_size_{gpu_tag}.png")