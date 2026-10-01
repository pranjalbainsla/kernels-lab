#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Thread block tile: one block computes a BM x BN tile of C, stepping along K in slabs of BK.
// Thread tile: each thread computes a TM x TN sub-tile of the block tile.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

__global__ void sgemm_2D_blocktile(int M, int K, int N, float alpha, const float *A, const float *B, float beta, float *C) {

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tileRow = blockIdx.x;
    const int tileCol = blockIdx.y;

    const int threadRow = threadIdx.x / (BN / TN);   // 0 .. BM/TM - 1
    const int threadCol = threadIdx.x % (BN / TN);   // 0 .. BN/TN - 1

    const int loadRowA = threadIdx.x / BK;
    const int loadColA = threadIdx.x % BK;
    const int loadRowB = threadIdx.x / BN;
    const int loadColB = threadIdx.x % BN;

    A += tileRow * BM * K;
    B += tileCol * BN;
    C += tileRow * BM * N + tileCol * BN;

    // Thread tile accumulators (TM x TN), kept in registers across all K-steps
    float acc[TM * TN] = {0.0f};
    // Register fragments: one column of the A slab and one row of the B slab
    float regA[TM] = {0.0f};
    float regB[TN] = {0.0f};

    for (int kTile = 0; kTile < K; kTile += BK) {
        // TODO: bounds checks; as written, M, N and K must be multiples of BM, BN and BK

        // Cooperative load of the A and B slabs into shared memory.
        // Each pass covers NUM_THREADS elements, so loop until the slab is covered.
        for (int loadOffset = 0; loadOffset < BM; loadOffset += NUM_THREADS / BK) {
            As[(loadRowA + loadOffset) * BK + loadColA] = A[(loadRowA + loadOffset) * K + loadColA];
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += NUM_THREADS / BN) {
            Bs[(loadRowB + loadOffset) * BN + loadColB] = B[(loadRowB + loadOffset) * N + loadColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            // Load this thread's fragments from shared memory into registers
            for (int i = 0; i < TM; ++i) {
                regA[i] = As[(threadRow * TM + i) * BK + k];
            }
            for (int j = 0; j < TN; ++j) {
                regB[j] = Bs[k * BN + threadCol * TN + j];
            }
            // Outer product of the two fragments, accumulated into the thread tile.
            // TM + TN shared loads feed TM * TN FMAs.
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i * TN + j] += regA[i] * regB[j];
                }
            }
        }

        // Barrier: don't let a fast thread overwrite the slabs for the next K-step
        // while slower threads are still reading them
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int row = threadRow * TM + i;
            int col = threadCol * TN + j;
            C[row * N + col] = alpha * acc[i * TN + j] + beta * C[row * N + col];
        }
    }
}

void launch_sgemm_2D_blocktile(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 block(NUM_THREADS);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    sgemm_2D_blocktile<<<grid, block, 0, stream>>>(M, K, N, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}