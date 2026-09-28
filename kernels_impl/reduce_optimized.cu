// warpReduce: does the final 32→1 reduction for a single warp, with NO
// __syncthreads() calls at all. This only works because all 32 threads here
// are in the SAME warp, and a warp executes in lockstep — every thread is
// guaranteed to be at the same instruction at the same time already, so
// there's nothing to synchronize.

// `template <unsigned int blockSize>` means blockSize is baked in at COMPILE
// time (you pick it when you instantiate the kernel, e.g. reduce6<256>).
// Because of this, every `if (blockSize >= 64)` below is decided by the
// compiler, not at runtime — branches that can never be true are deleted
// entirely from the compiled code. So for a fixed blockSize, this function
// becomes a straight-line sequence with no actual branching or looping.
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    // `volatile` tells the compiler: don't cache this value in a register,
    // always read/write it straight from shared memory. Without it, the
    // compiler might reorder or skip a read, assuming (wrongly) that no
    // other thread could have changed sdata since "last time" — but here,
    // other threads in the warp ARE changing it, one line at a time.

    // Each line halves the distance being combined: 32, 16, 8, 4, 2, 1.
    // Every thread with tid < that distance absorbs the value from
    // tid+distance. Because 32 threads run these lines in lockstep with no
    // sync needed, this whole unrolled sequence finishes in 6 instructions
    // instead of a 5-iteration loop with 5 __syncthreads() barriers.
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
    // Same shared-memory scratchpad idea as before: one slot per thread,
    // sized at launch time by the caller.
    extern __shared__ int sdata[];

    // tid = this thread's position within its block (used for shared mem).
    unsigned int tid = threadIdx.x;

    // Each block is responsible for a STARTING chunk of `2*blockSize`
    // elements (not just blockSize) — every thread grabs two elements up
    // front, see the "+= g_idata[i] + g_idata[i+blockSize]" line below.
    // This does the first level of addition during the LOAD itself, instead
    // of as a separate reduction step — half the work is already done by
    // the time the tree reduction below even starts.
    unsigned int i = blockIdx.x*(blockSize*2) + tid;

    // gridSize = total elements consumed per "round" by ALL blocks combined.
    // Used below to let each thread stride across the whole array, not just
    // its one starting chunk.
    unsigned int gridSize = blockSize*2*gridDim.x;

    // Start this thread's running total at zero.
    sdata[tid] = 0;

    // Grid-stride loop: if the input array is bigger than one full "pass"
    // across all blocks, each thread keeps striding forward by gridSize and
    // accumulating more elements into its own running total, instead of
    // launching more blocks than the GPU can run at once. This is what lets
    // one launch handle an array of ANY size, reusing the same threads.
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }

    // Barrier: every thread must finish accumulating its own total into
    // sdata before anyone starts combining values ACROSS threads below.
    __syncthreads();

    // Tree reduction from here down, same idea as the simpler reduce0, but:
    //  1. Contiguous threads stay active each pass (tid < half), not an
    //     interleaved/modulo pattern — so active threads pack into whole
    //     warps instead of scattering, avoiding the divergence problem.
    //  2. Each `if (blockSize >= X)` is again decided at compile time, so
    //     for a small blockSize, the unreachable stages compile away to
    //     nothing — you don't pay for reduction stages you don't need.

    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
        __syncthreads();
    }

    // Once we're down to 64 elements needing combination, everything left
    // fits inside a single warp (tid 0-31 does the work, tid 32-63 stops
    // mattering after this point). Hand off to the no-sync warp-level
    // version above — no more __syncthreads() needed from here on.
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    // Same as before: one thread per block writes out that block's total.
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}