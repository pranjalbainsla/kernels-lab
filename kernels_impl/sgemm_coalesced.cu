#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void sgemm_coalesced(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE); 
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  

  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      // A and B are row-major, so we need to index accordingly
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void launch_sgemm_coalesced(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  dim3 block(32 * 32);
  dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  sgemm_coalesced<<<grid, block, 0, stream>>>(M, K, N, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
