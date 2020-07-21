#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_1(int n,char x[n],char y[n])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= (n + 0) / 1) {
      x[1 * thread_x_id + -1] = y[1 * thread_x_id + -1];
    }
}

__global__ void _auto_kernel_0(int n,char y[n],char z[n])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= (n + 0) / 1) {
      y[1 * thread_x_id + -1] = z[1 * thread_x_id + -1];
    }
}

int main()
{
  int p;
  int n = 10;
  char x[n];
  char y[n];
  char z[n];
{
{
/* Auto-generated code for call to _auto_kernel_0 */
      typedef char _narray_y;
      _narray_y *d_y;
    cudaMalloc((void **) &d_y, sizeof(char ) * n);
    cudaMemcpy(d_y, y, sizeof(char ) * n, cudaMemcpyHostToDevice);
      typedef char _narray_z;
      _narray_z *d_z;
    cudaMalloc((void **) &d_z, sizeof(char ) * n);
    cudaMemcpy(d_z, z, sizeof(char ) * n, cudaMemcpyHostToDevice);
      int CUDA_GRID_X;
    CUDA_GRID_X = (1 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
      int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
      int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(n, d_y, d_z);
    cudaMemcpy(y, d_y, sizeof(char ) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(z, d_z, sizeof(char ) * n, cudaMemcpyDeviceToHost);
    }
{
/* Auto-generated code for call to _auto_kernel_1 */
      typedef char _narray_x;
      _narray_x *d_x;
    cudaMalloc((void **) &d_x, sizeof(char ) * n);
    cudaMemcpy(d_x, x, sizeof(char ) * n, cudaMemcpyHostToDevice);
      typedef char _narray_y;
      _narray_y *d_y;
    cudaMalloc((void **) &d_y, sizeof(char ) * n);
    cudaMemcpy(d_y, y, sizeof(char ) * n, cudaMemcpyHostToDevice);
      int CUDA_GRID_X;
    CUDA_GRID_X = (1 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
      int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
      int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_1<<<CUDA_gridSize,CUDA_blockSize>>>(n, d_x, d_y);
    cudaMemcpy(x, d_x, sizeof(char ) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, sizeof(char ) * n, cudaMemcpyDeviceToHost);
    }
  }
/*
	for(int p = 0; p < n; p++)
		y[p] = z[p];
	
	for(int p = 0; p < n; p++)
		x[p] = y[p];
*/
  return 0;
}
