# Notes

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