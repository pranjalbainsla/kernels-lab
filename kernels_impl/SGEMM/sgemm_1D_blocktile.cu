#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Thread block tile: one block computes a BM x BN tile of C, stepping along K in slabs of BK.
// Thread tile: each thread computes TM outputs (a TM x 1 column) of the block tile.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;

__global__ void sgemm_1D_blocktile(int M, int K, int N, float alpha, const float *A, const float *B, float beta, float *C) {

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tileRow = blockIdx.x;
    const int tileCol = blockIdx.y;

    const int threadRow = threadIdx.x / BN;   // 0 .. BM/TM - 1
    const int threadCol = threadIdx.x % BN;   // 0 .. BN - 1

    // Cooperative-load indices: each thread loads one float of the A slab and one of the B slab.
    const int loadRowA = threadIdx.x / BK;
    const int loadColA = threadIdx.x % BK;
    const int loadRowB = threadIdx.x / BN;
    const int loadColB = threadIdx.x % BN;

    A += tileRow * BM * K;
    B += tileCol * BN;
    C += tileRow * BM * N + tileCol * BN;

    float acc[TM] = {0.0f};

    // March along K one BK-wide slab at a time
    for (int kTile = 0; kTile < K; kTile += BK) {
        // TODO: bounds checks; as written, M, N and K must be multiples of BM, BN and BK

        As[loadRowA * BK + loadColA] = A[loadRowA * K + loadColA];
        Bs[loadRowB * BN + loadColB] = B[loadRowB * N + loadColB];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            // One Bs value, held in a register, is reused for all TM outputs.
            // Bs read: consecutive threadCol across a warp, so no bank conflicts.
            float b = Bs[k * BN + threadCol];
            for (int i = 0; i < TM; ++i) {
                // As read: same address across the warp (threadRow is uniform), so a broadcast.
                acc[i] += As[(threadRow * TM + i) * BK + k] * b;
            }
        }

        // Barrier: don't let a fast thread overwrite the slabs for the next K-step
        // while slower threads are still reading them
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int row = threadRow * TM + i;
        C[row * N + threadCol] = alpha * acc[i] + beta * C[row * N + threadCol];
    }
}

void launch_sgemm_1D_blocktile(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 block((BM * BN) / TM);   // 64*64/8 = 512 threads
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    sgemm_1D_blocktile<<<grid, block, 0, stream>>>(M, K, N, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}