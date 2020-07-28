#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_0(int a[100][2])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= 100 && thread_y_id <= 2) {
      a[2 * thread_x_id + -2][2 * thread_y_id + -2] = a[2 * thread_x_id + -1][2 * thread_y_id + -1];
    }
}

int main()
{
  int a[100][2];
  int i;
  int j;
  i = 0;
{
{
/* Auto-generated code for call to _auto_kernel_0 */
      typedef int _narray_a[2];
      _narray_a *d_a;
    cudaMalloc((void **) &d_a, sizeof(int ) * 100 * 2);
    cudaMemcpy(d_a, a, sizeof(int ) * 100 * 2, cudaMemcpyHostToDevice);
      int CUDA_GRID_X;
    CUDA_GRID_X = (100 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
      int CUDA_GRID_Y;
    CUDA_GRID_Y = (2 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
      int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(d_a);
    cudaMemcpy(a, d_a, sizeof(int ) * 100 * 2, cudaMemcpyDeviceToHost);
    }
  }
  return 2;
}
