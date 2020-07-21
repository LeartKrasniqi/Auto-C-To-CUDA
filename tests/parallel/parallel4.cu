#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_1(int a[10][10],int b[10][10],int ecs_serial_index_0)
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= ecsMinFn(1 + (ecs_serial_index_0 - 1) * 3 + (3 - 1),8) && thread_y_id <= 10) {
      a[1 * thread_y_id + 3][1 * thread_x_id + 4] = b[1 * thread_y_id + 0][1 * thread_x_id + 0];
      b[1 * thread_y_id + 2][1 * thread_x_id + 3] = a[1 * thread_y_id + 0][1 * thread_x_id + 0];
    }
}

__global__ void _auto_kernel_0(int a[10][10],int b[10][10],int ecs_serial_index_0)
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= ecsMinFn(1 + (ecs_serial_index_0 - 1) * 2 + (2 - 1),10) && thread_y_id <= 8) {
      a[1 * thread_x_id + 3][1 * thread_y_id + 4] = b[1 * thread_x_id + 0][1 * thread_y_id + 0];
      b[1 * thread_x_id + 2][1 * thread_y_id + 3] = a[1 * thread_x_id + 0][1 * thread_y_id + 0];
    }
}

__device__ __host__ int ecsMinFn(int val1,int val2)
{
  if (val1 > val2) 
    return val2;
   else 
    return val1;
}

__device__ __host__ int ecsMaxFn(int val1,int val2)
{
  if (val1 > val2) 
    return val1;
   else 
    return val2;
}

int agon()
{
  int j;
  int i;
  int a[10][10];
  int b[10][10];
{
{
/* Auto-generated code for calls to _auto_kernel_0 to _auto_kernel_1 */
      typedef int _narray_a[10];
      _narray_a *d_a;
    cudaMalloc((void **) &d_a, sizeof(int ) * 10 * 10);
    cudaMemcpy(d_a, a, sizeof(int ) * 10 * 10, cudaMemcpyHostToDevice);
      typedef int _narray_b[10];
      _narray_b *d_b;
    cudaMalloc((void **) &d_b, sizeof(int ) * 10 * 10);
    cudaMemcpy(d_b, b, sizeof(int ) * 10 * 10, cudaMemcpyHostToDevice);
      for (int ecs_serial_index_0 = 1; ecs_serial_index_0 <= 3; ++ecs_serial_index_0) {{
          int CUDA_GRID_X;
    CUDA_GRID_X = (10 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
          int CUDA_GRID_Y;
    CUDA_GRID_Y = (10 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
          int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(d_a, d_b, ecs_serial_index_0);
        }
{
          int CUDA_GRID_X;
    CUDA_GRID_X = (10 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
          int CUDA_GRID_Y;
    CUDA_GRID_Y = (10 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
          int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_1<<<CUDA_gridSize,CUDA_blockSize>>>(d_a, d_b, ecs_serial_index_0);
        }
      }
    cudaMemcpy(a, d_a, sizeof(int ) * 10 * 10, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, sizeof(int ) * 10 * 10, cudaMemcpyDeviceToHost);
    }
  }
  return 100;
}
