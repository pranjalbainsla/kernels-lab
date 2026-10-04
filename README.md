# PLOTS

### 1) Parallel reduction

Sum-reduction of 2^22 int32 elements (16 MiB) with 256 threads per block on a Tesla T4 (236 GB/s measured memory bandwidth)

| Kernel | Time (ms) | Bandwidth (GB/s) | Step speedup | Cumulative |
|---|:---:|:---:|:---:|:---:|
| Kernel 1: interleaved (divergent) | 0.4963 | 33.8 |  |  |
| Kernel 2: interleaved (bank conflicts) | 0.3004 | 55.9 | 1.65x | 1.65x |
| Kernel 3: sequential addressing | 0.2592 | 64.7 | 1.16x | 1.91x |
| Kernel 4: first add during load | 0.2031 | 82.6 | 1.28x | 2.44x |
| Kernel 5: unroll last warp | 0.0916 | 183.1 | 2.22x | 5.42x |
| Kernel 6: completely unrolled | 0.1092 | 153.7 | 0.84x | 4.55x |
| Kernel 7: multiple elements/thread | 0.0693 | 242.1 | 1.58x | 7.16x |

### 2) SGEMM 
<p align="center">
  <img src="./plots/sgemm_vs_size_Tesla_T4.png" alt="SGEMM GFLOPS/s vs matrix size">
</p>

