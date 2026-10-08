# kernel optimizations

Hand-written CUDA kernels, each optimized step by step and benchmarked against a hardware ceiling or a library baseline (cuBLAS). Everything below was measured on a Tesla T4.

## Results

### Parallel reduction
- 2^28 int32 elements, 256 threads/block
- Measured peak bandwidth: 278.17 GB/s (theoretical peak: 320 GB/s)

| Kernel | Time (ms) | Bandwidth (GB/s) | Step speedup | Cumulative |
|---|:---:|:---:|:---:|:---:|
| 1: interleaved (divergent) | 17.1257 | 62.7 |  |  |
| 2: interleaved (bank conflicts) | 13.2998 | 80.7 | 1.29x | 1.29x |
| 3: sequential addressing | 10.7794 | 99.6 | 1.23x | 1.59x |
| 4: first add during load | 6.1437 | 174.8 | 1.75x | 2.79x |
| 5: unroll last warp | 4.8255 | 222.5 | 1.27x | 3.55x |
| 6: completely unrolled | 4.8757 | 220.2 | 0.99x | 3.51x |
| 7: multiple elements/thread | 4.2383 | 253.3 | 1.15x | 4.04x |

> Final kernel reaches 253.3 GB/s, about 91% of the measured peak (read-only) bandwidth ceiling 

### SGEMM
- M = K = N = 4096, alpha = 1, beta = 0
- Each kernel is timed against cuBLAS in the same run

| # | Kernel | Limiter it addressed | GFLOPS | % of cuBLAS |
|---|---|---|---|---|
| 1 | Naive | (baseline; uncoalesced global loads) | 61.7 | 1.7% |
| 2 | Coalesced | wasted sectors per request | 540.7 | 14.4% |
| 3 | Shared-memory tiling | no data reuse across threads (AI 0.25 -> 8) | 890.0 | 22.6% |
| 4 | 1D block-tiling | shared-memory loads per FMA (MIO stalls) | 1548.1 | 40.0% |
| 5a | 2D block-tiling, 64×64 | shared-memory loads per FMA (2.0 → 0.25) | 2171.7 | 54.3% |
| 5b | 2D block-tiling, 128×128 | global traffic per FLOP, warps per block, sync overhead (AI 16 → 32) | 2950.3 | 76.6% |

<p align="center">
  <img src="./plots/sgemm_vs_size_Tesla_T4.png" alt="SGEMM GFLOPS vs matrix size">
</p>

## Where things are

| Path | What it is |
|---|---|
| [kernels_impl/reduction/](kernels_impl/reduction) | `reduce1.cu` … `reduce7.cu`, each one adds one optimization to the previous |
| [kernels_impl/SGEMM/](kernels_impl/SGEMM) | naive → coalesced → shared mem → 1D blocktile → 2D blocktile |
| [reduce_report.py](reduce_report.py) | Builds and runs all reduction kernels and prints the table above |
| [run.py](run.py) | Builds one SGEMM kernel, checks it against `torch.matmul`, times it vs cuBLAS |
| [plots/bench_plot_sgemm.py](plots/bench_plot_sgemm.py) | Sweeps SGEMM kernels over matrix sizes and produces the plot |
| [kernels_impl/reduction/bw_ceiling.cu](kernels_impl/reduction/bw_ceiling.cu) | Bandwidth ceiling (roof) for the reduction kernels. This roof must use the same array size, block/grid size and timing loop as the reduce kernels to be a fair comparison |
| [gpu_props.cu](gpu_props.cu) | Prints device properties (SM count, smem, registers) |
| [notes.md](notes.md) | Roofline math and analysis notes |

## Reproduce

```bash
python reduce_report.py --n 28     # reduction table
python run.py --kernel sgemm_naive --M 4096 --N 4096 --K 4096
python plots/bench_plot_sgemm.py   # SGEMM plot
```

Requires an NVIDIA GPU, CUDA toolkit, and PyTorch. Results will vary by GPU and clocks; the numbers above are from a single T4 session.

## References
- Reduction: Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA)

