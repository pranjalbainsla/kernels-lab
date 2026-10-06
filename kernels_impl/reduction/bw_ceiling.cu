#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdint>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " at line " << __LINE__ << std::endl; \
        std::exit(1); \
    } } while (0)

// BANDWIDTH CEILING: read the whole array once, do (almost) nothing with it.
// Same data size, block size, grid size and timing loop as the reduce kernels,
// so the result is the roof they should be compared against.
__global__ void read_only(const int4 *in, int *out, int n4) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    int acc = 0;
    for (; i < n4; i += gridSize) {
        int4 v = in[i];                 // 16 B per thread, fully coalesced
        acc ^= v.x ^ v.y ^ v.z ^ v.w;   // cheap use so the loads are not optimized away
    }
    // One write per thread would add traffic; only write if something is non-zero (keeps acc alive).
    if (acc == 0x7fffffff) out[0] = acc;
}

#ifndef LOG2N
#define LOG2N 22  // override with nvcc -DLOG2N=<k>
#endif

int main() {
    const int32_t n = 1 << LOG2N;
    const size_t bytes = n * sizeof(int);
    const int n4 = n / 4;

    const int iters = 100;
    constexpr unsigned int blockSize = 256;

    std::vector<int> host_in(n);
    srand(42);
    for (int &x : host_in) x = rand() % 100;

    int *dev_in, *dev_out;
    CHECK(cudaMalloc(&dev_in, bytes));
    CHECK(cudaMalloc(&dev_out, sizeof(int)));
    CHECK(cudaMemcpy(dev_in, host_in.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dev_out, 0, sizeof(int)));

    int sms;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int num_blocks = std::min((n4 + blockSize - 1) / blockSize, sms * 32);

    // Warm-up
    read_only<<<num_blocks, blockSize>>>(reinterpret_cast<const int4 *>(dev_in), dev_out, n4);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Timed runs with CUDA events
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        read_only<<<num_blocks, blockSize>>>(reinterpret_cast<const int4 *>(dev_in), dev_out, n4);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaGetLastError());

    float total_ms;
    CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double ms = total_ms / iters;

    // Same output format as the reduce kernels so reduce_report.py's regexes work on it.
    std::cout << "Avg time: " << ms << " ms\n";
    std::cout << "Effective bandwidth: " << bytes / ms / 1e6 << " GB/s\n";

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dev_in));
    CHECK(cudaFree(dev_out));
}
