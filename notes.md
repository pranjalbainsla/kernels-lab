# Notes
**Ceilings** 
```text
Tesla T4: Measured bandwidth: 236.0 GB/s  (300-320 theoretical).
Tesla T4: Measured FP32 throughput: 4.21 TFLOPS (8.1 theo). 
Ridge point: 18 FLOP/byte
```
> Ridge point is essentially the arithmetic intensity at which the memory roof and the compute roof meet. A kernel with AI below the ridge is limited by memory, and above it by compute.

(M = K = N = 4096 as example):
```text
FLOPs = 2MNK + MN = 137.4 GFLOP
Minimum DRAM bytes = 4 * (MK + KN + MN) = 201 MB (A, B read once, C written once)
DRAM-ideal intensity = 137.4e9 / 201e6 = 683 FLOP/byte
Compute floor = 137.4 GFLOP / 4.21 TFLOPS = 32.6 ms
Memory floor = 201 MB / 236 GB/s = 0.85 ms
```
note: DRAM stops being a non-issue once traffic exceeds ~38x the ideal 201 MB (=7.6 GB), because the memory time then exceeds the compute floor (32.6 ms).
___

### 1) Naive SGEMM

- **Mapping:** one thread per element of C (4096^2 threads). Each thread reads one row of A, one column of B, and writes one element of C.
- **Traffic with no caching:** (2·4096 + 1) · 4096² · 4 B = 550 GB, vs 201 MB ideal.
- **Traffic reported by the profiler:** 17.9 GB of DRAM reads, so the caches absorb most of the 550 GB. Even so, this exceeds the 38x threshold, and 17.9 GB / 236 GB/s = 76 ms of DRAM time alone is above the compute floor.
- **Arithmetic intensity:** 2 flop per 8 bytes loaded = 0.25 FLOP/byte, far left of the ridge point (18 FLOP/byte), so let's focus on memory first.

```text
    GPU: Tesla T4
    sgemm_naive  M=4092 N=4092 K=4092  alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_naive :    846.66 ms      161.9 GFLOPS
    cuBLAS      :     33.71 ms     4065.5 GFLOPS
    sgemm_naive is 4.0% of cuBLAS
```

> **Limiter:** poor global memory access pattern. DRAM traffic is far above ideal, and the kernel makes 4.3 G global requests that mostly hit L1 and L2 instead of reusing data in shared memory.
___

### 2) **Global memory coalescing**
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
___

### 3) **Shared memory cache blocking**
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
Let's do some occupancy calc now. To do that, we need to find what the kernel is limited by (shared memory/SM, the number of threads per block, the number of registers per thread). That'll give you the upper limit of how many block you can load per SM. Final occupancy can then be calculated as num active warps / max active warps per multiprocessor.

| Property                       | Value        | Property                               | Value        |
|--------------------------------|--------------|----------------------------------------|--------------|
| Name                           | Tesla T4     | max regs per multiprocessor            | 65536        |
| Compute Capability             | 7.5          | reg allocation unit size               | 256 (table)  |
| max threads per block          | 1024         | reg allocation granularity             | warp (table) |
| max threads per multiprocessor | 1024         | total global mem                       | 14912 MB     |
| threads per warp               | 32           | max shared mem per block               | 48 KB        |
| warp allocation granularity    | 4 (table)    | CUDA runtime shared mem overhead/block | 0 B (table)  |
| max regs per block             | 65536        | shared mem per multiprocessor          | 65536 B      |
| multiprocessor count           | 40           | max warps per multiprocessor           | 32           |

- Shared memory: (65536B per SM) / (8192B per Block) = 8 Blocks upper limit
- Threads: 1024 Threads per Block, max 1024 threads per SM => Upper limit 1 block
- Registers: 39 regs per thread * 32 threads per warp = 1248 regs per warp. Register allocation granularity is 256 regs on a warp level, hence rounding up to 1280 regs per warp. We have 32 warps per block, so 40960 regs per block. Max 65536 regs per SM => upper limit 1 block

So, we're limited by the number of threads per block, and the number of registers per thread. The theoretical occupancy is 100%. So, of that isnt a problem, there is probably some stalls happening

As per the warp state statistics profile,
> On average, each warp of this workload spends 27.3 cycles being stalled waiting for the MIO (memory input/output) instruction queue to be not full.

The inner loop does 2 shared-memory loads (As, Bs) for every 1 FMA. The warps flood the queue with loads while the FMA units sit mostly idle. So, we could try to do more FMAs per shared-memory load.
___

### 4) **1-D blocktiling**
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
**Memory accesses per result:**

(1 result per thread, 32*32 tile)
- GMEM: the outer loop runs K/32 times, with 2 loads => K/16 per thread
- SMEM: each outer iteration has an inner loop of 32 steps, and each step reads 1 As and 1 Bs value (2 loads) => K/32 × 32 × 2 = 2K

(TM=8 results per thread, 64*64 tile, BK=8)
- GMEM: the outer loop runs K/8 times, with 2 loads each => K/4 per thread. The thread makes 8 results, so per result it's K/32. (*2X fewer*)
- SMEM: each outer iteration has 8 inner steps, and each step reads 1 Bs value (held in a register and reused) plus 8 As values, so 9 loads. That's K/8 × 8 × 9 = 9K per thread. Per result it's 9K ÷ 8 = 9K/8. (*1.8X fewer*)

> On average, each warp of this workload spends 6.4 cycles being stalled waiting for the MIO (memory input/output) instruction queue to be not full.  

(clearly an improvement)
___
### 5) **2D blocktiling** 

From kernel 3 to kernel 4, theoretical occupancy dropped from 100% to 50%, yet performance still improved. Higher occupancy isn't always faster, you only need enough resident warps to hide memory latency.

From kernel 4 to kernel 5, I kept the thread block size the same but made each thread compute a larger tile (8x more outputs per thread). This left only 2 warps per block. Performance still improved because arithmetic intensity went up: each byte loaded from memory is reused for more FLOPs, so stalls on memory matter less and low occupancy hurts less.

Next, I increased the block tile size, which gave 8 warps per block and doubled arithmetic intensity again (16 to 32 FLOP/byte).

- **BM=BN=64, TM=TN=8**: intensity = 2·BM·BN / (4·(BM+BN)) = 16 FLOP/byte

```text
    GPU: Tesla T4
    ptxas info    : Used 123 registers, used 1 barriers, 4096 bytes smem, 400 bytes cmem[0]
    sgemm_2D_blocktile  M=4096 K=4096 N=4096 alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_2D_blocktile:     60.42 ms     2274.6 GFLOPS
    cuBLAS      :     32.29 ms     4255.9 GFLOPS
    sgemm_2D_blocktile is 53.4% of cuBLAS
```
- **BM=BN=128, TM=TN=8**: intensity = 2·BM·BN / (4·(BM+BN)) = 32 FLOP/byte
```text
    GPU: Tesla T4
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    ptxas info    : Used 123 registers, used 1 barriers, 8192 bytes smem, 400 bytes cmem[0]
    sgemm_2D_blocktile  M=4096 K=4096 N=4096 alpha=1.0 beta=0.0
    correct: True  (max abs err = 0.000e+00)
    sgemm_2D_blocktile:     43.00 ms     3196.0 GFLOPS
    cuBLAS      :     32.24 ms     4262.9 GFLOPS
    sgemm_2D_blocktile is 75.0% of cuBLAS
```