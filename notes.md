# notes

The goal is to write a performant CUDA SGEMM (C = αAB + βC, single precision) from scratch, starting from a naive kernel and optimizing one bottleneck at a time (on a tesla T4).

A is M×K, B is K×N, C is M×N. All matrices are row-major. Benchmark shape: M = K = N = 4096.


## Back-of-the-envelope

Before writing any kernel, spec numbers tell us which regime we are in.

Spec peaks are never reached in practice (a 70 W T4 throttles clocks and power), so the roofs below use **measured, achievable** numbers:

- **Achievable compute:** cuBLAS SGEMM on the same shape, ~**4 TFLOPS** (3.8–4.2 across runs). Spec is 8.1 TFLOPS.
- **Achievable bandwidth:** a copy kernel (`bw_ceiling.cu`), **278 GB/s**. Spec is 320 GB/s.
- **Ridge point** = achievable FLOPS / achievable bandwidth = 4000 GFLOPS / 278 GB/s ≈ **14.4 FLOP/byte** (the spec numbers would give 25.3).
- A kernel whose arithmetic intensity (AI, FLOPs per byte moved from DRAM) is below the ridge is **memory-bound**; above it, **compute-bound**. That tells us whether to fix data movement or instruction throughput.

```text
FLOPs                = 2·M·K·N = 137.4 GFLOP    (one multiply + one add per k-step, the standard GEMM convention)
Minimum DRAM traffic = 4·(MK + KN + 2MN) = 268 MB  (read A, B, C once; write C once)
Ideal AI             = 137.4 GFLOP / 268 MB ≈ 512 FLOP/byte
Compute floor        = 137.4 GFLOP / 4 TFLOPS    ≈ 34 ms   (spec 8.1 TFLOPS would give 17 ms)
Memory floor         = 268 MB / 278 GB/s         ≈ 0.96 ms (spec 320 GB/s would give 0.84 ms)
Ridge point          = 4000 GFLOPS / 278 GB/s    ≈ 14.4 FLOP/byte
```

Ideal AI (512) is far right of the ridge, so a nicely written SGEMM is firmly compute-bound. Every kernel below is therefore a story about how much of that ideal reuse we actually capture. Naive kernels re-fetch the same data over and over, so their *effective* AI is tiny and they behave as memory-bound.

> note: Each kernel is reported as "% of cuBLAS", timed in the same `run.py` run so both see the same clocks.

**A caveat on AI.** The AI figures below assume every tile load goes to DRAM. In reality L2 absorbs some repeated loads, so treat AI as an upper bound on DRAM traffic. The profiler os what has the final say on what is actually limiting.

(TODO: record driver and library versions, and note what each affects.)


## Kernel 1: Naive

**Mapping:** one thread per element of C (4096^2 threads). Each thread reads one row of A and one column of B, and writes one element of C.

```text
GPU: Tesla T4
sgemm_naive  M=4096 K=4096 N=4096 alpha=1.0 beta=0.0
sgemm_naive :   2228.95 ms       61.7 GFLOPS
cuBLAS      :     36.85 ms     3730.0 GFLOPS
sgemm_naive is 1.7% of cuBLAS
```

**Arithmetic intensity.** Every k-step loads 8 bytes (one float from A, one from B) for 2 FLOPs, so AI = 0.25 FLOP/byte, far left of the 14.4 ridge (memory-bound)

> Limiter: a poor global-memory access pattern (the profiler reports 16.5 sectors/request).

Some vocabulary: a *request* is one warp-level memory instruction (32 threads loading together). Memory is served in 32 B *sectors*, and a 128 B cache line is 4 sectors. If each thread wants 4 B, the best case is 32 × 4 B = 128 B in one cache line, i.e. **4 sectors/request**; the worst case is every thread landing in a different sector, i.e. **32 sectors/request**.

Whether we get the best or worst case depends on how consecutive threads (consecutive `threadIdx.x`) map to elements of C. Take three neighboring threads at the same k-step:

- **Case 1: consecutive threads differ in the row index** (C[i][j], C[i+1][j], C[i+2][j]). They load A[i][k], A[i+1][k], A[i+2][k], which are K floats apart in a row-major matrix, so each lands in a different sector. **Uncoalesced and expensive** (up to 32 sectors/request). They all load the same B[k][j], a cheap broadcast.
- **Case 2: consecutive threads differ in the column index** (C[i][j], C[i][j+1], C[i][j+2]). They load B[k][j], B[k][j+1], B[k][j+2], which sit side by side, so the loads **coalesce** into a few sectors. They all load the same A[i][k], a broadcast costing one sector. **Cheap.**

The naive kernel maps `threadIdx.x` to the row index (Case 1). The fix is to swap the mapping so `threadIdx.x` indexes the column (Case 2).


## Kernel 2: Global memory coalescing

```text
sgemm_coalesced:    254.19 ms      540.7 GFLOPS
cuBLAS         :     36.70 ms     3745.2 GFLOPS
sgemm_coalesced is 14.4% of cuBLAS
```

- Sectors/request dropped from 16.5 to 2.5 (yay)

Coalescing makes each byte fetched from memory useful, but it does not reduce how many bytes each thread asks for. Each thread still loads 8 B per 2 FLOPs, so AI is still 0.25 FLOP/byte and we are still memory-bound. To raise AI we need threads to **share** loaded data, and the thing threads in a block can share is **shared memory**.


## Kernel 3: Shared-memory tiling

The idea is that each block cooperatively loads a tile of A and a tile of B into shared memory, then every thread computes from the tile, so each global load is reused by many threads.

With a tile edge T (here 32), a block loads two T×T tiles (2·T²·4 bytes) per step and does 2·T³ FLOPs on them:

AI = 2·T³ / (2·T²·4) = T/4 = **8 FLOP/byte** for T = 32 (32x better than before).

```text
sgemm_smem  :    154.43 ms      890.0 GFLOPS
cuBLAS      :     34.90 ms     3938.0 GFLOPS
sgemm_smem is 22.6% of cuBLAS
```

Compiling with `--ptxas-options=-v`:

```text
ptxas info    : Used 39 registers, used 1 barriers, 8192 bytes smem, 400 bytes cmem[0]
```

AI went up 32x but speed only went up 1.5x, so something else is limiting us. AI = 8 is still below the 14.4 ridge, which would cap a DRAM-limited kernel at about 8 × 278 ≈ 2.2 TFLOPS, yet we are at 0.82. So DRAM is not what is binding (L2 is absorbing repeated tile loads). Is the GPU even full of work? Let's check occupancy: how many blocks fit per SM, limited by whichever resource runs out first.

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

- **Shared memory:** 65536 B per SM / 8192 B per block = 8 blocks.
- **Threads:** 32×32 = 1024 threads per block, and an SM holds at most 1024 threads, so 1 block.
- **Registers:** 39 regs × 32 threads = 1248 per warp, rounded up to the 256-register allocation unit = 1280. × 32 warps = 40960 per block. 65536 / 40960 = 1.6, so 1 block.

The binding limit is 1 block per SM, which is 32 warps out of a maximum of 32: *100% theoretical occupancy. So the SM is not starved for warps. Warps are waiting on something else. The warp-state statistics say:

> On average, each warp of this workload spends 25.9 cycles being stalled waiting for the MIO (memory input/output) instruction queue to be not full.

The MIO queue handles shared-memory instructions. Looking at the inner loop, each FMA needs one `As` load and one `Bs` load from shared memory: 2 shared-memory loads per FMA. The warps flood the queue with loads while the FMA units sit mostly idle. We moved the bottleneck from DRAM to shared memory. The next step is to do **more FMAs per shared-memory load**.


## Kernel 4: 1D block-tiling

The idea is to let each thread compute TM = 8 outputs in a column instead of 1. In the inner loop, a thread loads one `Bs` value into a register and reuses it for 8 FMAs against 8 different `As` values.

Configuration: 64×64 block tile, BK = 8, so a block has 64·64/8 = 512 threads.

```text
ptxas info    : Used 68 registers, used 1 barriers, 4096 bytes smem, 400 bytes cmem[0]
sgemm_1D_blocktile:     88.78 ms     1548.1 GFLOPS
cuBLAS      :     35.55 ms     3865.7 GFLOPS
sgemm_1D_blocktile is 40.0% of cuBLAS
```

**Global AI.** For a BM×BN block tile, AI = 2·BM·BN / (4·(BM + BN)) (BK cancels). For 64×64 that is 16 FLOP/byte, 2x the 32×32 kernel, and already above the 14.4 ridge. From here on, DRAM bandwidth should no longer be the limiter, the remaining stalls are on-chip (shared memory, instruction issue).

**Memory accesses, per result computed:**

| | Kernel 3 (1 result/thread, 32×32 tile) | Kernel 4 (8 results/thread, 64×64 tile, BK = 8) |
|---|---|---|
| GMEM loads per thread | K/32 outer iterations × 2 = K/16 | K/8 iterations × 2 = K/4, i.e. **K/32 per result** (2x fewer) |
| SMEM loads per thread | K/32 iterations × 32 steps × 2 = 2K | K/8 iterations × 8 steps × (1 Bs + 8 As = 9) = 9K, i.e. **9K/8 per result** (1.8x fewer) |

The MIO stall dropped accordingly:

> On average, each warp of this workload spends 6.4 cycles being stalled waiting for the MIO (memory input/output) instruction queue to be not full.

One side effect worth noting: with 68 registers per thread and 512 threads per block, only 1 block fits per SM, so theoretical occupancy fell from 100% to 50% (16 of 32 warps). Yet performance doubled. Occupancy is only a means of hiding latency: we need *enough* resident warps, not the maximum. Reducing shared-memory traffic per FMA mattered more.

We are still loading 9 values for 8 FMAs, about 1.1 shared loads per FMA. If each thread computed a 2D patch, that ratio would improve much further.


## Kernel 5: 2D block-tiling

Now, let's make each thread computes a TM×TN = 8×8 patch of C. Per inner step it loads 8 values of `As` and 8 values of `Bs` into registers (16 shared loads) and performs an **outer product** of 64 FMAs: **0.25 shared loads per FMA**, down from around 1.1 in kernel 4 and 2.0 in kernel 3.

**First attempt: BM = BN = 64, TM = TN = 8.** The block tile is the same size as in kernel 4, but each thread now does 8x more work, so the block shrinks from 512 threads to 64 threads (only 2 warps per block).

AI = 2·64·64 / (4·(64 + 64)) = 16 FLOP/byte, unchanged from kernel 4

```text
ptxas info    : Used 123 registers, used 1 barriers, 4096 bytes smem, 400 bytes cmem[0]
sgemm_2D_blocktile:     63.29 ms     2171.7 GFLOPS
cuBLAS      :     34.39 ms     3997.0 GFLOPS
sgemm_2D_blocktile is 54.3% of cuBLAS
```

The speedup over kernel 4 comes from the **shared-memory-to-register reuse**, not from global AI, which did not change. It works despite very few warps per block because each thread has 64 independent FMAs in flight, so there is plenty of instruction-level parallelism to cover latency.

**Second attempt: BM = BN = 128, TM = TN = 8.** A bigger block tile halves the global traffic per FLOP, and brings the block back to 256 threads (8 warps).

AI = 2·128·128 / (4·(128 + 128)) = 32 FLOP/byte

```text
ptxas info    : Used 123 registers, used 1 barriers, 8192 bytes smem, 400 bytes cmem[0]
sgemm_2D_blocktile:     46.58 ms     2950.3 GFLOPS
cuBLAS      :     35.70 ms     3849.3 GFLOPS
sgemm_2D_blocktile is 76.6% of cuBLAS
```

---

To try next: vectorized loads, avoiding shared-memory bank conflicts, warp-level tiling, and double buffering.