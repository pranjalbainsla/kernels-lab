#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

__global__ void sgemm_2D_blocktile(int M, int K, int N, float alpha, const float *A, const float *B, float beta, float *C) {

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    const int threadRow = threadIdx.x / (BN/TN);   // 0..(BM/TM - 1)
    const int threadCol = threadIdx.x % (BN/TN);   // 0..(BN/TN - 1)

    // indices for loading A (BM x BK) and B (BK x BN) into shared memory
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += NUM_THREADS/BK) {
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += NUM_THREADS/BN) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // load relevant As & Bs entries into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            // perform outer product on register cache, 
            // accumulate into threadResults
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int row = threadRow * TM + resIdxM;
            int col = threadCol * TN + resIdxN;
            C[row * N + col] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[row * N + col];
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