/* Simple benchmark to test matrix addition */
#include <stdio.h>
#include <math.h>
#define ROWS 100000
#define COLS 100000
#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_0(float A[100000][100000],float B[100000][100000],float C[100000][100000])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= 100000 && thread_y_id <= 100000) {
      C[1 * thread_x_id + -1][1 * thread_y_id + -1] = A[1 * thread_x_id + -1][1 * thread_y_id + -1] + B[1 * thread_x_id + -1][1 * thread_y_id + -1];
    }
}

int main()
{
  int j_nom_4;
  int i_nom_3;
  int j_nom_2;
  int i_nom_1;
  int j;
  int i;
/* Declare three arrays: C = A + B */
  float A[100000][100000];
  float B[100000][100000];
  float C[100000][100000];
/* Initialize */
  for (i = 1; i <= 100000; i += 1) {
    for (j = 1; j <= 100000; j += 1) {
      A[1 * i + -1][1 * j + -1] = (sin((1 * i + -1 + (1 * j + -1))) * sin((1 * i + -1 + (1 * j + -1))));
      B[1 * i + -1][1 * j + -1] = (cos((1 * i + -1 + (1 * j + -1))) * cos((1 * i + -1 + (1 * j + -1))));
    }
  }
{
/* Auto-generated code for call to _auto_kernel_0 */
    typedef float _narray_A[100000];
    _narray_A *d_A;
    cudaMalloc((void **) &d_A, sizeof(float ) * 100000 * 100000);
    cudaMemcpy(d_A, A, sizeof(float ) * 100000 * 100000, cudaMemcpyHostToDevice);
    typedef float _narray_B[100000];
    _narray_B *d_B;
    cudaMalloc((void **) &d_B, sizeof(float ) * 100000 * 100000);
    cudaMemcpy(d_B, B, sizeof(float ) * 100000 * 100000, cudaMemcpyHostToDevice);
    typedef float _narray_C[100000];
    _narray_C *d_C;
    cudaMalloc((void **) &d_C, sizeof(float ) * 100000 * 100000);
    cudaMemcpy(d_C, C, sizeof(float ) * 100000 * 100000, cudaMemcpyHostToDevice);
    int CUDA_GRID_X;
    CUDA_GRID_X = (100000 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
    int CUDA_GRID_Y;
    CUDA_GRID_Y = (100000 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
    int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(d_A, d_B, d_C);
    cudaMemcpy(A, d_A, sizeof(float ) * 100000 * 100000, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, sizeof(float ) * 100000 * 100000, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, d_C, sizeof(float ) * 100000 * 100000, cudaMemcpyDeviceToHost);
  }
/* Check the result */
  double sum = 0;
  for (i_nom_3 = 1; i_nom_3 <= 100000; i_nom_3 += 1) {
    for (j_nom_4 = 1; j_nom_4 <= 100000; j_nom_4 += 1) {
      sum += C[1 * i_nom_3 + -1][1 * j_nom_4 + -1];
    }
  }
/* Report the result */
  double r = (double )100000;
  double c = (double )100000;
  printf("Result (Should be close to 1.00) : %f\n",sum / (r * c));
  return 0;
}
