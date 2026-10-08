#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Tile edge: one thread block computes one TILE x TILE tile of C.
// Block size is TILE * TILE threads, flattened to 1D.
#define TILE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void sgemm_smem(int M, int K, int N, float alpha, const float *A, const float *B, float beta, float *C) {

  // Shared-memory tiles of A and B, reused by all threads in the block
  __shared__ float As[TILE * TILE];
  __shared__ float Bs[TILE * TILE];

  const int tileRow = blockIdx.x;
  const int tileCol = blockIdx.y;

  const int threadRow = threadIdx.x / TILE;
  const int threadCol = threadIdx.x % TILE;

  const int x = tileRow * TILE + threadRow;  // row of C
  const int y = tileCol * TILE + threadCol;  // col of C

  A += tileRow * TILE * K;
  B += tileCol * TILE;
  C += tileRow * TILE * N + tileCol * TILE;

  // Accumulator lives in a register across all K-steps
  float tmp = 0.0f;

  for (int kTile = 0; kTile < K; kTile += TILE) {
    const int aCol = kTile + threadCol;  // col in A (K dimension)
    const int bRow = kTile + threadRow;  // row in B (K dimension)

    // each thread copies one element of the A tile and one of the B tile
    // from global memory into shared memory
    As[threadRow * TILE + threadCol] = (x < M && aCol < K) ? A[threadRow * K + threadCol] : 0.0f;
    Bs[threadRow * TILE + threadCol] = (bRow < K && y < N) ? B[threadRow * N + threadCol] : 0.0f;

    __syncthreads(); // so all warps finish writing to the shared memory before any warp starts reading

    A += TILE;
    B += TILE * N;

    // Each thread computes a partial dot product from shared memory
    for (int k = 0; k < TILE; ++k) {
      tmp += As[threadRow * TILE + k] * Bs[k * TILE + threadCol];
    }

    // Barrier: don't let a fast thread overwrite the shared tiles
    // for the next K-step while slower threads are still reading them
    __syncthreads();
  }

  if (x < M && y < N) {
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
  }
}

void launch_sgemm_smem(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  dim3 block(TILE * TILE);
  dim3 grid(CEIL_DIV(M, TILE), CEIL_DIV(N, TILE));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  sgemm_smem<<<grid, block, 0, stream>>>(M, K, N, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}