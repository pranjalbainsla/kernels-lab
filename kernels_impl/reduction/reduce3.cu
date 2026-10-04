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

// REDUCTION 3
__global__ void reduce3(int *in, int *out, int n){
    extern __shared__ int sdata[];  // dynamic shared memory, sized to blockDim.x at launch

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < n) sdata[tid] = in[i] + ((i + blockDim.x < n) ? in[i + blockDim.x] : 0);
    else sdata[tid] = 0;

    __syncthreads();

    // Reduction method -> occurs in shared memory
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }
    if (tid == 0){
        atomicAdd(out, sdata[0]); // writes the partial sum back as one indivisible operation, so no other thread can interleave in the middle.
    }
}


int main() {
    // random fun fact: The C++ standard only guarantees at least 16 bits for int, 
    // but we want to be sure that we have 32 bits.
    const int32_t n = 1 << 22;
    // size_t matches the machine's address width
    // so it's 32 bits on 32-bit systems and 64 bits on 64-bit systems
    const size_t bytes = n * sizeof(int);

    const int iters = 100;
    const int blockSize = 256;  // runtime value; the kernel reads it from blockDim.x

    // Host data
    std::vector<int> host_in(n); // frees memory automatically when it goes out of scope
    srand(42);
    for (int &x : host_in) x = rand() % 100;

    // Device data 
    int *dev_in, *dev_out; // declare two pointers to int that will hold GPU (device) memory addresses
    CHECK(cudaMalloc(&dev_in, bytes));
    CHECK(cudaMalloc(&dev_out, sizeof(int)));
    CHECK(cudaMemcpy(dev_in, host_in.data(), bytes, cudaMemcpyHostToDevice));

    int num_blocks = (n + blockSize - 1) / blockSize;
    num_blocks = (num_blocks + 1) / 2; // since each block processes two elements per thread

    // Warm-up (excludes context/launch overhead from timing)
    CHECK(cudaMemset(dev_out, 0, sizeof(int)));
    reduce3<<<num_blocks, blockSize, blockSize * sizeof(int)>>>(dev_in, dev_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Timed runs with CUDA events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaMemset(dev_out, 0, sizeof(int)));
        reduce3<<<num_blocks, blockSize, blockSize * sizeof(int)>>>(dev_in, dev_out, n);
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