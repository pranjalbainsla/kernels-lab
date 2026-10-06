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
| [ceilings.py](ceilings.py) | Measures the GPU's achievable memory bandwidth and FP32 throughput (roofline ceilings) |
| [kernels_impl/reduction/bw_ceiling.cu](kernels_impl/reduction/bw_ceiling.cu) | Bandwidth ceiling (roof) for the reduction kernels. Kept separate from ceilings.py because reduction only reads, while ceilings.py measures a PyTorch copy (read + write), and this roof must use the same array size, block/grid size and timing loop as the reduce kernels to be a fair comparison |
| [gpu_props.cu](gpu_props.cu) | Prints device properties (SM count, smem, registers) |
| [notes.md](notes.md) | Roofline math and analysis notes |

## Reproduce

```bash
python ceilings.py                 # hardware ceilings
python reduce_report.py --n 28     # reduction table
python run.py --kernel sgemm_naive --M 4096 --N 4096 --K 4096
python plots/bench_plot_sgemm.py   # SGEMM plot
```

Requires an NVIDIA GPU, CUDA toolkit, and PyTorch. Results will vary by GPU and clocks; the numbers above are from a single T4 session.

## References
- Reduction: Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA)

