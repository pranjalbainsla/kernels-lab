# PLOTS

### 1) Parallel reduction

Number of Elements: 2^28 int32
Number of Threads per Block: 256


| Kernel | Time (ms) | Bandwidth (GB/s) | Step speedup | Cumulative |
|---|:---:|:---:|:---:|:---:|
| Kernel 1: interleaved (divergent) | 18.4270 | 58.3 |  |  |
| Kernel 2: interleaved (bank conflicts) | 14.4062 | 74.5 | 1.28x | 1.28x |
| Kernel 3: sequential addressing | 12.0862 | 88.8 | 1.19x | 1.52x |
| Kernel 4: first add during load | 6.6745 | 160.9 | 1.81x | 2.76x |
| Kernel 5: unroll last warp | 5.3795 | 199.6 | 1.24x | 3.43x |
| Kernel 6: completely unrolled | 4.6441 | 231.2 | 1.16x | 3.97x |
| Kernel 7: multiple elements/thread | 4.2645 | 251.8 | 1.09x | 4.32x |

### 2) SGEMM 
<p align="center">
  <img src="./plots/sgemm_vs_size_Tesla_T4.png" alt="SGEMM GFLOPS/s vs matrix size">
</p>

