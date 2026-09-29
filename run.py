"""Utility runner: compiles a kernel from kernels_impl/<name>.cu, checks it against
torch.matmul, and times it against cuBLAS.

Examples:
    python run.py  # sgemm_naive, 4096^3
    python run.py --kernel sgemm_naive --M 1024 --N 2048 --K 512
    python run.py --alpha 2.0 --beta 0.5 --iters 10
"""
import argparse
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

KERNEL_DIR = Path(__file__).parent / "kernels_impl"


def build(kernel: str, verbose: bool = False):
    """Compile kernels_impl/<kernel>.cu and return the module holding launch_<kernel>."""
    src_path = KERNEL_DIR / f"{kernel}.cu"
    if not src_path.exists():
        available = sorted(p.stem for p in KERNEL_DIR.glob("*.cu"))
        raise SystemExit(f"No kernel '{kernel}'. Available: {available}")
    fn = f"launch_{kernel}"
    decl = (
        f"void {fn}(torch::Tensor A, torch::Tensor B, torch::Tensor C, "
        f"float alpha, float beta);"
    )
    return getattr(
        load_inline(
            name=f"{kernel}_ext",
            cpp_sources=decl,
            cuda_sources=src_path.read_text(),
            functions=[fn],
            extra_cuda_cflags=["--ptxas-options=-v"] if verbose else [],
            verbose=verbose,
        ),
        fn,
    )


def time_ms(fn, warmup, iters):
    """Median GPU time in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", default="sgemm_naive")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5, help="timed runs of your kernel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", type=bool, default=False)
    args = p.parse_args()

    assert torch.cuda.is_available(), "No GPU found"
    torch.backends.cuda.matmul.allow_tf32 = False  # fair FP32 cuBLAS baseline
    print("GPU:", torch.cuda.get_device_name(0))

    launch = build(args.kernel, verbose=args.verbose)
    M, K, N = args.M, args.K, args.N

    torch.manual_seed(args.seed)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    C0 = C.clone()

    # correctness
    launch(A, B, C, args.alpha, args.beta)
    torch.cuda.synchronize()
    ref = args.alpha * (A @ B) + args.beta * C0
    max_err = (C - ref).abs().max().item()
    ok = torch.allclose(C, ref, rtol=1e-3, atol=1e-2)
    print(f"{args.kernel}  M={M} K={K} N={N} alpha={args.alpha} beta={args.beta}")
    print(f"correct: {ok}  (max abs err = {max_err:.3e})")

    # timing (values in C drift if beta != 0, which doesn't affect timing)
    flops = 2 * M * N * K
    t_mine = time_ms(lambda: launch(A, B, C, args.alpha, args.beta), args.warmup, args.iters)
    t_ref = time_ms(lambda: torch.matmul(A, B), args.warmup, 20)

    g_mine = flops / (t_mine * 1e-3) / 1e9
    g_ref = flops / (t_ref * 1e-3) / 1e9
    print(f"{args.kernel:<12}: {t_mine:9.2f} ms  {g_mine:9.1f} GFLOPS")
    print(f"{'cuBLAS':<12}: {t_ref:9.2f} ms  {g_ref:9.1f} GFLOPS")
    print(f"{args.kernel} is {100 * g_mine / g_ref:.1f}% of cuBLAS")


if __name__ == "__main__":
    main()