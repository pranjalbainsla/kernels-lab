#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;  // row of C
  const int y = blockIdx.y * blockDim.y + threadIdx.y;  // col of C

  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      # A and B are row-major, so we need to index accordingly
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void launch_sgemm_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  dim3 block(32, 32);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  sgemm_naive<<<grid, block, 0, stream>>>(M, N, K, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
