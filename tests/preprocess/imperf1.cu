#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_2(int a[5][5],int b[5][5],int i)
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= 5 && thread_y_id <= 5) {
      b[i][1 * thread_y_id + -1] = a[i][1 * thread_y_id + -1];
    }
}

__global__ void _auto_kernel_1(int b[5][5],int i)
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= 5) {
      b[i][0] = 1;
    }
}

__global__ void _auto_kernel_0(int a[5][5])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= 5) {
      a[1 * thread_x_id + -1][1 * thread_x_id + -1] = 1;
    }
}

int main()
{
  int j;
  int i_nom_2;
  int i_nom_1;
  int i;
  int a[5][5];
  int b[5][5];
  int y;
{
{
/* Auto-generated code for call to _auto_kernel_0 */
      typedef int _narray_a[5];
      _narray_a *d_a;
    cudaMalloc((void **) &d_a, sizeof(int ) * 5 * 5);
    cudaMemcpy(d_a, a, sizeof(int ) * 5 * 5, cudaMemcpyHostToDevice);
      int CUDA_GRID_X;
    CUDA_GRID_X = (5 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
      int CUDA_GRID_Y;
    CUDA_GRID_Y = (5 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
      int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(d_a);
    cudaMemcpy(a, d_a, sizeof(int ) * 5 * 5, cudaMemcpyDeviceToHost);
    }
{
/* Auto-generated code for call to _auto_kernel_1 */
      typedef int _narray_b[5];
      _narray_b *d_b;
    cudaMalloc((void **) &d_b, sizeof(int ) * 5 * 5);
    cudaMemcpy(d_b, b, sizeof(int ) * 5 * 5, cudaMemcpyHostToDevice);
      int CUDA_GRID_X;
    CUDA_GRID_X = (5 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
      int CUDA_GRID_Y;
    CUDA_GRID_Y = (5 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
      int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_1<<<CUDA_gridSize,CUDA_blockSize>>>(d_b, i);
    cudaMemcpy(b, d_b, sizeof(int ) * 5 * 5, cudaMemcpyDeviceToHost);
    }
{
/* Auto-generated code for call to _auto_kernel_2 */
      typedef int _narray_a[5];
      _narray_a *d_a;
    cudaMalloc((void **) &d_a, sizeof(int ) * 5 * 5);
    cudaMemcpy(d_a, a, sizeof(int ) * 5 * 5, cudaMemcpyHostToDevice);
      typedef int _narray_b[5];
      _narray_b *d_b;
    cudaMalloc((void **) &d_b, sizeof(int ) * 5 * 5);
    cudaMemcpy(d_b, b, sizeof(int ) * 5 * 5, cudaMemcpyHostToDevice);
      int CUDA_GRID_X;
    CUDA_GRID_X = (5 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
      int CUDA_GRID_Y;
    CUDA_GRID_Y = (5 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
      int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_2<<<CUDA_gridSize,CUDA_blockSize>>>(d_a, d_b, i);
    cudaMemcpy(a, d_a, sizeof(int ) * 5 * 5, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, sizeof(int ) * 5 * 5, cudaMemcpyDeviceToHost);
    }
  }
/* This should NOT be convertible */
  for (int i = 0; i < 5; i++) {
    a[i][i] = b[i][i];
    for (int j = 0; j < 5; j++) 
      b[i][j] = a[i][j];
  }
  return 0;
}
