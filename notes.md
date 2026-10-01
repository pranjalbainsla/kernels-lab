# Notes
**Ceilings** 
```text
Tesla T4: Measured bandwidth: 236.0 GB/s  (300-320 theoretical).
Tesla T4: Measured FP32 throughput: 4.21 TFLOPS (8.1 theo). 
Ridge point: 17 FLOP/byte
```
> Ridge point is essentially the arithmetic intensity at which the memory roof and the compute roof meet. A kernel with AI below the ridge is limited by memory, and above it by compute.

(M = K = N = 4096 as example):
```text
FLOPs = 2MNK + MN = 137.4 GFLOP
Minimum DRAM bytes = 4 * (MK + KN + MN) = 201 MB (A, B read once, C written once)
DRAM-ideal intensity = 137.4e9 / 201e6 = 683 FLOP/byte
Floor time (compute) = FLOPs / cuBLAS GFLOPS = 32.6 ms
```
note: DRAM ideal intensity is far right of our ridge point, meaning DRAM roof will never be the reason our SGEMM kernel being slow. The question will always be which inner resource (L1/LSU, smem, FMA pipe, latency) runs out first. To simplify, ideally memory transfers = 201 / 236 = 0.85 ms, which is much smaller than the floor time

1) **Naive SGEMM**
One thread per element of C, 4096^2 threads in total, each loads one row of matrix A and one col of matrix B and one element of C.
Memory traffic = $((2*4096 + 1)*4096^2) * 4$ = 550 GB! (compare with 201 MB ideal)

- Arithmetic Intensity = 2 flops per 8 bytes = 0.25 flop/byte (vs our 17flop/byte)

```text
    GPU: Tesla T4
    sgemm_naive  M=4092 N=4092 K=4092  alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_naive :    846.66 ms      161.9 GFLOPS
    cuBLAS      :     33.71 ms     4065.5 GFLOPS
    sgemm_naive is 4.0% of cuBLAS
```
> Limiter: poor global memory access pattern

2) **Global memory coalescing**
Global memory is fetched in 32B "sectors" (a 128 B cache line is 4 sectors). When a warp's 32 floats are consecutive and aligned, the 128 B load is served by just 4 sectors; scattered addresses touch more sectors (up to 32), wasting bandwidth.

```text
    GPU: Tesla T4
    sgemm_coalesced  M=4092 K=4092 N=4092 alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_coalesced:    253.52 ms      540.5 GFLOPS
    cuBLAS      :     34.74 ms     3944.4 GFLOPS
    sgemm_coalesced is 13.7% of cuBLAS
```
Per-warp iteration:
- A load = 4 bytes utilised (broadcasted) | 1 sector 
- B load = 128 bytes utilised | 4 sectors
**Bytes/sector:** 132/5 = 26.4 (compared to 4 bytes/sector in naive kernel)
> Note: we're still at 2 flops/byte but the change in thread->output mapping helps us minimise GMEM accesses.

3) **Shared memory cache blocking**
```text
    GPU: Tesla T4
    sgemm_smem  M=4096 K=4096 N=4096 alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_smem  :    147.91 ms      929.2 GFLOPS
    cuBLAS      :     32.02 ms     4292.8 GFLOPS
    sgemm_smem is 21.6% of cuBLAS
```
For a tile edge of 32:
AI = $2*32^3 / 2*32^2*4$ = 8 flop/byte

Compiling with --ptxas-options=-v:  
```text
ptxas info    : Used 39 registers, used 1 barriers, 8192 bytes smem, 400 bytes cmem[0]
ptxas info    : Compile time = 77.160 ms
```
TODO: Find what the kernel is limited by (shared memory/SM, the number of threads per block, the number of registers per thread). That'll give you the upper limit of how many block you can load per SM. Final occupancy can then be calculated as num active warps / max active warps per multiprocessor

4) 1-D blocktiling
```text
    GPU: Tesla T4

    ptxas info    : Used 68 registers, used 1 barriers, 4096 bytes smem, 400 bytes cmem[0]
    ptxas info    : Compile time = 72.129 ms

    sgemm_1D_tile  M=4096 K=4096 N=4096 alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_1D_tile:     82.71 ms     1661.7 GFLOPS
    cuBLAS      :     34.44 ms     3991.2 GFLOPS
    sgemm_1D_tile is 41.6% of cuBLAS
```

5) 2D blocktiling
```text
    GPU: Tesla T4

    ptxas info    : Used 123 registers, used 1 barriers, 4096 bytes smem, 400 bytes cmem[0]
    ptxas info    : Compile time = 103.930 ms

    sgemm_2D_blocktile  M=4096 K=4096 N=4096 alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_2D_blocktile:     60.42 ms     2274.6 GFLOPS
    cuBLAS      :     32.29 ms     4255.9 GFLOPS
    sgemm_2D_blocktile is 53.4% of cuBLAS
```
