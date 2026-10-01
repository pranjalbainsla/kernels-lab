// Usage (Colab): !nvcc gpu_props.cu -o gpu_props && ./gpu_props
#include <cstdio>
#include <cuda_runtime.h>

// Per-compute-capability values NOT exposed by cudaDeviceProp (from NVIDIA occupancy calculator tables)
struct ArchInfo {
    int warpAllocGranularity;   // warps
    int regAllocUnitSize;       // registers
    const char* regAllocGran;   // "warp" or "block"
    int sharedMemOverheadPerBlock; // bytes
};

static ArchInfo getArchInfo(int major, int minor) {
    int cc = major * 10 + minor;
    ArchInfo a;
    a.warpAllocGranularity = (cc >= 80) ? 4 : (cc >= 50 ? 4 : 2);
    a.regAllocUnitSize = (cc == 70 || cc == 72 || cc == 75 || cc >= 80) ? 256 : 256;
    a.regAllocGran = "warp";
    a.sharedMemOverheadPerBlock = (cc >= 80) ? 1024 : 0;
    return a;
}

int main() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    ArchInfo a = getArchInfo(p.major, p.minor);

    printf("Name\t\t\t\t\t%s\n", p.name);
    printf("Compute Capability\t\t\t%d.%d\n", p.major, p.minor);
    printf("max threads per block\t\t\t%d\n", p.maxThreadsPerBlock);
    printf("max threads per multiprocessor\t\t%d\n", p.maxThreadsPerMultiProcessor);
    printf("threads per warp\t\t\t%d\n", p.warpSize);
    printf("warp allocation granularity\t\t%d (table)\n", a.warpAllocGranularity);
    printf("max regs per block\t\t\t%d\n", p.regsPerBlock);
    printf("max regs per multiprocessor\t\t%d\n", p.regsPerMultiprocessor);
    printf("reg allocation unit size\t\t%d (table)\n", a.regAllocUnitSize);
    printf("reg allocation granularity\t\t%s (table)\n", a.regAllocGran);
    printf("total global mem\t\t\t%zu MB\n", p.totalGlobalMem >> 20);
    printf("max shared mem per block\t\t%zu KB\n", p.sharedMemPerBlock >> 10);
    printf("CUDA runtime shared mem overhead/block\t%d B (table)\n", a.sharedMemOverheadPerBlock);
    printf("shared mem per multiprocessor\t\t%zu B\n", p.sharedMemPerMultiprocessor);
    printf("multiprocessor count\t\t\t%d\n", p.multiProcessorCount);
    printf("max warps per multiprocessor\t\t%d\n", p.maxThreadsPerMultiProcessor / p.warpSize);
    return 0;
}
