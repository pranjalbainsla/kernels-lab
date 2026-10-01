__device__ int warpReduce(int val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;   // full sum ends up in lane 0
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;

    // Each block is responsible for `2*blockSize` elements
    // Each thread grabs two elements up front (performs the first add during the load!)
    unsigned int i = blockIdx.x*(blockSize*2) + tid;

    // stride across the whole array to handle arrays larger than one full "pass" across all blocks
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
 
    while (i < n){ 
        sdata[tid] += g_idata[i] + g_idata[i+blockSize]; 
        i += gridSize; 
    }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) {
        int v = sdata[tid] + sdata[tid + 32];
        v = warpReduce(v);
        if (tid == 0) g_odata[blockIdx.x] = v;
    }

}