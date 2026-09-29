# Notes
**Ceilings** 
```text
Tesla T4: Measured bandwidth: 236.0 GB/s  (300-320 theoretical).
Tesla T4: Measured FP32 throughput: 4.21 TFLOPS (8.1 theo). 
```

1) Naive SGEMM
```text
    GPU: Tesla T4
    sgemm_naive  M=4092 N=4092 K=4092  alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_naive :    846.66 ms      161.9 GFLOPS
    cuBLAS      :     33.71 ms     4065.5 GFLOPS
    sgemm_naive is 4.0% of cuBLAS
```

2) Global memory coalescing
```text
    GPU: Tesla T4
    sgemm_coalesced  M=4092 K=4092 N=4092 alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_coalesced:    253.52 ms      540.5 GFLOPS
    cuBLAS      :     34.74 ms     3944.4 GFLOPS
    sgemm_coalesced is 13.7% of cuBLAS
```

3) Shared memory cache blocking
```text
    GPU: Tesla T4
    sgemm_smem  M=4096 K=4096 N=4096 alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_smem  :    147.91 ms      929.2 GFLOPS
    cuBLAS      :     32.02 ms     4292.8 GFLOPS
    sgemm_smem is 21.6% of cuBLAS
```
Compiling with --ptxas-options=-v:  
```text
ptxas info    : Used 39 registers, used 1 barriers, 8192 bytes smem, 400 bytes cmem[0]
ptxas info    : Compile time = 77.160 ms
```
TODO: Find what the kernel is limited by (shared memory/SM, the number of threads per block, the number of registers per thread). That'll give you the upper limit of how many block you can load per SM. Final occupancy can then be calculated as num active warps / max active warps per multiprocessor