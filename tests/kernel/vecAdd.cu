#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_0(int a[100000],int b[100000],int c[100000])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= 100000) {
      c[1 * thread_x_id + -1] = a[1 * thread_x_id + -1] + b[1 * thread_x_id + -1];
    }
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main()
{
  int a[100000];
  int b[100000];
  int c[100000];
  int i = 0;
  for (i = 1; i <= 100000; i += 1) {
    a[1 * i + -1] = (sin((1 * i + -1)) * sin((1 * i + -1)));
    b[1 * i + -1] = (cos((1 * i + -1)) * cos((1 * i + -1)));
  }
{
/* Auto-generated code for call to _auto_kernel_0 */
    typedef int _narray_a;
    _narray_a *d_a;
    cudaMalloc((void **) &d_a, sizeof(int ) * 100000);
    cudaMemcpy(d_a, a, sizeof(int ) * 100000, cudaMemcpyHostToDevice);
    typedef int _narray_b;
    _narray_b *d_b;
    cudaMalloc((void **) &d_b, sizeof(int ) * 100000);
    cudaMemcpy(d_b, b, sizeof(int ) * 100000, cudaMemcpyHostToDevice);
    typedef int _narray_c;
    _narray_c *d_c;
    cudaMalloc((void **) &d_c, sizeof(int ) * 100000);
    cudaMemcpy(d_c, c, sizeof(int ) * 100000, cudaMemcpyHostToDevice);
    int CUDA_GRID_X;
    CUDA_GRID_X = (100000 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
    int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
    int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(d_a, d_b, d_c);
    cudaMemcpy(a, d_a, sizeof(int ) * 100000, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, sizeof(int ) * 100000, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, sizeof(int ) * 100000, cudaMemcpyDeviceToHost);
  }
  double sum = 0;
  for (i = 1; i <= 100000; i += 1) {
    sum += c[1 * i + -1];
  }
  printf("Final result (Should be close to 1): %f\n",sum);
  return 0;
}
