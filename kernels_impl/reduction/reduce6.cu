#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <cstdint>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " at line " << __LINE__ << std::endl; \
        std::exit(1); \
    } } while (0)

// REDUCTION 5
__device__ int warpReduce(int val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;   // full sum ends up in lane 0
}
template <unsigned int blockSize>
__global__ void reduce6(int *in, int *out, int n){
    __shared__ int sdata[blockSize];  // stored in the shared memory; size known at compile time

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;

    if (i < n) sdata[tid] = in[i] + ((i + blockSize < n) ? in[i + blockSize] : 0);
    else sdata[tid] = 0;

    __syncthreads();

    // blockSize is a compile-time constant, so this loop is fully unrolled
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (blockSize >= 64) { if (tid < 32) sdata[tid] += sdata[tid + 32]; __syncthreads(); }
    
    if (tid < 32) {
        int v = warpReduce(sdata[tid]);
        if(tid == 0) atomicAdd(out, v); // writes the partial sum back as one indivisible operation, so no other thread can interleave in the middle.
    }
}


#ifndef LOG2N
#define LOG2N 22  // override with nvcc -DLOG2N=<k>
#endif

int main() {
    // random fun fact: The C++ standard only guarantees at least 16 bits for int, 
    // but we want to be sure that we have 32 bits.
    const int32_t n = 1 << LOG2N;
    // values in [0, MAX_VAL): keeps the int32 sum from overflowing, so the CPU/GPU check is meaningful
    constexpr int MAX_VAL = 8;
    static_assert((int64_t)(MAX_VAL - 1) * (1LL << LOG2N) <= INT32_MAX, "sum would overflow int32; lower MAX_VAL or LOG2N");
    // size_t matches the machine's address width
    // so it's 32 bits on 32-bit systems and 64 bits on 64-bit systems
    const size_t bytes = n * sizeof(int);

    const int iters = 100;
    constexpr unsigned int blockSize = 256;  // compile-time: passed to the kernel as a template parameter

    // Host data
    std::vector<int> host_in(n); // frees memory automatically when it goes out of scope
    srand(42);
    for (int &x : host_in) x = rand() % MAX_VAL;

    // Device data 
    int *dev_in, *dev_out; // declare two pointers to int that will hold GPU (device) memory addresses
    CHECK(cudaMalloc(&dev_in, bytes));
    CHECK(cudaMalloc(&dev_out, sizeof(int)));
    CHECK(cudaMemcpy(dev_in, host_in.data(), bytes, cudaMemcpyHostToDevice));

    int num_blocks = (n + blockSize - 1) / blockSize;
    num_blocks = (num_blocks + 1) / 2; // since each block processes two elements per thread

    // Warm-up (excludes context/launch overhead from timing)
    CHECK(cudaMemset(dev_out, 0, sizeof(int)));
    reduce6<blockSize><<<num_blocks, blockSize>>>(dev_in, dev_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Timed runs with CUDA events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaMemset(dev_out, 0, sizeof(int)));
        reduce6<blockSize><<<num_blocks, blockSize>>>(dev_in, dev_out, n);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaGetLastError());

    float total_ms;
    CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double ms = total_ms / iters;

    // Result (the last run left the final sum in dev_out)
    int gpuResult;
    CHECK(cudaMemcpy(&gpuResult, dev_out, sizeof(int), cudaMemcpyDeviceToHost));

    // CPU verification
    const int cpuResult = std::accumulate(host_in.begin(), host_in.end(), 0);
    const bool ok = (gpuResult == cpuResult);

    std::cout << (ok ? "\033[32m" : "\033[31m")
              << (ok ? "Verification successful" : "Verification failed")
              << ": GPU = " << gpuResult << ", CPU = " << cpuResult
              << "\033[0m\n";
    std::cout << "Avg time: " << ms << " ms\n";
    std::cout << "Effective bandwidth: " << bytes / ms / 1e6 << " GB/s\n";

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dev_in));
    CHECK(cudaFree(dev_out));
}