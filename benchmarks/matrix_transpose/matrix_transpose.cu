/* Simple program to obtain transpose of a matrix */
#include <stdio.h>
#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_0(int m,int n,char A[m][n],char B[n][m])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= (m + 0) / 1 && thread_y_id <= (n + 0) / 1) {
      B[1 * thread_y_id + -1][1 * thread_x_id + -1] = A[1 * thread_x_id + -1][1 * thread_y_id + -1];
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
  int m = 1000;
  int n = 500;
  char A[m][n];
  char B[n][m];
/* Initialize */
  srand((time(0)));
  for (i = 1; i <= (m + 0) / 1; i += 1) {
    for (j = 1; j <= (n + 0) / 1; j += 1) {
      A[1 * i + -1][1 * j + -1] = (rand() % 25 + 'A');
    }
  }
{
/* Auto-generated code for call to _auto_kernel_0 */
    typedef char _narray_A[n];
    _narray_A *d_A;
    cudaMalloc((void **) &d_A, sizeof(char ) * m * n);
    cudaMemcpy(d_A, A, sizeof(char ) * m * n, cudaMemcpyHostToDevice);
    typedef char _narray_B[m];
    _narray_B *d_B;
    cudaMalloc((void **) &d_B, sizeof(char ) * n * m);
    cudaMemcpy(d_B, B, sizeof(char ) * n * m, cudaMemcpyHostToDevice);
    int CUDA_GRID_X;
    CUDA_GRID_X = (1 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
    int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
    int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(m, n, d_A, d_B);
    cudaMemcpy(A, d_A, sizeof(char ) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, sizeof(char ) * n * m, cudaMemcpyDeviceToHost);
  }
/* Assertion */
  for (i_nom_3 = 1; i_nom_3 <= (m + 0) / 1; i_nom_3 += 1) {
    for (j_nom_4 = 1; j_nom_4 <= (n + 0) / 1; j_nom_4 += 1) {
      if (A[1 * i_nom_3 + -1][1 * j_nom_4 + -1] != B[1 * j_nom_4 + -1][1 * i_nom_3 + -1]) {
        fprintf(stderr,"ERROR\n");
        exit(-1);
      }
    }
  }
  printf("All good b0ss\n");
  return 0;
}
