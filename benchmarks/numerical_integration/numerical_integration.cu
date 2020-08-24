/* Benchmark that calculate the integral of F(x) over the interval [A,B] */
#include <stdio.h>
#define NUM_INTERVALS 1000000
#define F(x) (x)*(x)
#define A 0
#define B 10
#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_0(float arr[1000000],float delta)
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= 1000000) {
      float x = ((float )0) + ((float )(1 * thread_x_id + -1)) * delta;
      arr[1 * thread_x_id + -1] = x * x + x * delta * (x * delta);
    }
}

int main()
{
  int i_nom_1;
  int i;
/* Array to hold answer */
  float arr[1000000];
/* Step size */
  float delta = ((float )10) / ((float )1000000);
{
/* Auto-generated code for call to _auto_kernel_0 */
    typedef float _narray_arr;
    _narray_arr *d_arr;
    cudaMalloc((void **) &d_arr, sizeof(float ) * 1000000);
    cudaMemcpy(d_arr, arr, sizeof(float ) * 1000000, cudaMemcpyHostToDevice);
    int CUDA_GRID_X;
    CUDA_GRID_X = (1000000 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
    int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
    int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(d_arr, delta);
    cudaMemcpy(arr, d_arr, sizeof(float ) * 1000000, cudaMemcpyDeviceToHost);
  }
/* Add up the heights of the rectangles */
  double sum = 0;
  for (i_nom_1 = 1; i_nom_1 <= 1000000; i_nom_1 += 1) {
    sum += arr[1 * i_nom_1 + -1];
  }
/* Multiply by width of rectangle */
  sum *= delta / 2.0;
/* Report result */
  printf("Result: %f\n",sum);
  return 0;
}
