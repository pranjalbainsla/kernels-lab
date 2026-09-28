// Sums all the values in one block into a single number, using shared memory.
// Each block produces one partial sum; you'd sum the per-block results afterward
// (either on the CPU, or with another kernel launch).

// This kernel uses "interleaved addressing" to sum the values in a block.
__global__ void reduce0(int *g_idata, int *g_odata) {
    // Shared memory scratchpad for this block, sized at launch time by the
    // caller (the "extern" + no size here means the size is passed when the
    // kernel is launched, e.g. reduce0<<<blocks, threads, threads*sizeof(int)>>>).
    // Every thread in the block can read/write this — it's the fast, on-chip
    // memory shared within a block, not global DRAM.
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    // tid = this thread's position within its own block (0 to blockDim.x-1).
    // Used to index into shared memory, which is local to the block.
    unsigned int tid = threadIdx.x;

    // i = this thread's global index across the whole grid, same formula as
    // any other kernel: which block it's in, times block size, plus position
    // within the block. Used to index into the actual input array in DRAM.
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    // Each thread copies exactly one element from slow global memory (g_idata)
    // into the block's fast shared memory (sdata), at its own slot.
    sdata[tid] = g_idata[i];

    // Barrier: every thread in the block must reach this point before any
    // thread is allowed to continue. Necessary here because the reduction
    // loop below has threads reading slots that OTHER threads just wrote —
    // without this, some threads could start reading sdata before their
    // neighbor has finished writing it.
    __syncthreads();

    // do reduction in shared mem
    // Halves the number of "active" additions each pass: s = 1, 2, 4, 8, ...
    // This is a tree-shaped sum: pass 1 combines pairs 1 apart, pass 2 combines
    // results 2 apart, pass 3 combines results 4 apart, and so on, until only
    // sdata[0] holds the total for the whole block.
    for(unsigned int s=1; s < blockDim.x; s *= 2) {

        // Only threads whose tid is a multiple of (2*s) do work this pass —
        // e.g. pass 1 (s=1): tid 0,2,4,6... add their right neighbor.
        // Pass 2 (s=2): tid 0,4,8... add the neighbor 2 slots over. And so on.
        // This is "interleaved addressing" — the active threads are spread out
        // (every other one, then every 4th, ...) rather than packed together.
        // Note for later: this scatters which threads in a warp are active,
        // which causes warp divergence and is why this is the SLOW, naive
        // version of reduction (later versions fix exactly this).
        if (tid % (2*s) == 0) {
            // This thread absorbs its neighbor's value into its own slot.
            sdata[tid] += sdata[tid + s];
        }

        // Barrier again: must wait for every thread's addition in this pass
        // to finish before starting the next pass, since the next pass reads
        // values this pass just wrote.
        __syncthreads();
    }

    // write result for this block to global mem
    // After the loop, sdata[0] holds the sum of everything this block loaded.
    // Only one thread (tid == 0) needs to write it out — every other thread
    // would just be writing the same value redundantly.
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// int main() {
//     // Normal C++ mixed with CUDA API
//     std::cout << "Running CUDA from C++" << std::endl;
//     // ... allocate memory with cudaMalloc, launch kernel, etc.
//     return 0;
// }
// Compile with nvcc main.cu -o cuda_app