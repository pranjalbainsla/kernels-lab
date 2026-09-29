#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;

__global__ void sgemm_1D_blocktile(int M, int K, int N, float alpha, const float *A, const float *B, float beta, float *C) {

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    const int threadRow = threadIdx.x / BN;   // 0..(BM/TM - 1)
    const int threadCol = threadIdx.x % BN;   // 0..(BN - 1)

    // indices for loading A (BM x BK) and B (BK x BN) into shared memory,
    // one float per thread per load, threads flattened over blockDim.x
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float Btmp = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        int row = threadRow * TM + resIdx;
        C[row * N + threadCol] = alpha * threadResults[resIdx] + beta * C[row * N + threadCol];
    }
}

void launch_sgemm_1D_blocktile(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 block((BM * BN) / TM);   // e.g. 64*64/8 = 512 threads
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    sgemm_1D_blocktile<<<grid, block, 0, stream>>>(M, K, N, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}