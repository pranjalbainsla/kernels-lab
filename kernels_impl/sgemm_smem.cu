#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void sgemm_smem(int M, int K, int N, float alpha, const float *A, const float *B, float beta, float *C) {

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];   
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / BLOCKSIZE; // threadRow = threadIdx.y if we didnt flatten the block into 1D (for global memory coalescing)
  const int threadCol = threadIdx.x % BLOCKSIZE; 

  A += cRow * BLOCKSIZE * K;                    
  B += cCol * BLOCKSIZE;                        
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float tmp = 0.0;
  
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // TODO: add boundary checks to avoid out-of-bounds memory access

    // Have each thread load one of the elements in A & B from
    // global memory into shared memory.
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
        tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }

  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

void launch_sgemm_smem(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  sgemm_smem<<<grid, block, 0, stream>>>(M, K, N, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
