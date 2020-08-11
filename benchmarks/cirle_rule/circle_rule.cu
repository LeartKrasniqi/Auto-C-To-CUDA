/* Computes quadrature rules (i.e. circumference) for unit circle in 2D */
/* Adapted from: https://people.sc.fsu.edu/~jburkardt/c_src/circle_rule/circle_rule.html */
#define NUM_ANGLES 1000
#define PI 3.14159265358
#define F(x,y) x*y
#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_1(float w[1000],float Q[1000],float x[1000],float y[1000])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= 1000) {
      Q[1 * thread_x_id + -1] = w[1 * thread_x_id + -1] * x[1 * thread_x_id + -1] * y[1 * thread_x_id + -1];
    }
}

__global__ void _auto_kernel_0(float w[1000],float a[1000])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= 1000) {
      w[1 * thread_x_id + -1] = ((float )(1.0 / ((double )((float )1000))));
      a[1 * thread_x_id + -1] = ((float )(6.28319 * ((double )((float )(1 * thread_x_id + -1))) / ((double )((float )1000))));
    }
}

int main()
{
  int i_nom_3;
  int i_nom_2;
  int i_nom_1;
  int i;
/* Weights */
  float w[1000];
/* Angles */
  float a[1000];
/* Result */
  float Q[1000];
{
/* Auto-generated code for call to _auto_kernel_0 */
    typedef float _narray_w;
    _narray_w *d_w;
    cudaMalloc((void **) &d_w, sizeof(float ) * 1000);
    cudaMemcpy(d_w, w, sizeof(float ) * 1000, cudaMemcpyHostToDevice);
    typedef float _narray_a;
    _narray_a *d_a;
    cudaMalloc((void **) &d_a, sizeof(float ) * 1000);
    cudaMemcpy(d_a, a, sizeof(float ) * 1000, cudaMemcpyHostToDevice);
    int CUDA_GRID_X;
    CUDA_GRID_X = (1000 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
    int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
    int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(d_w, d_a);
    cudaMemcpy(w, d_w, sizeof(float ) * 1000, cudaMemcpyDeviceToHost);
    cudaMemcpy(a, d_a, sizeof(float ) * 1000, cudaMemcpyDeviceToHost);
  }
/* Useful sin/cos values */
  float x[1000];
  float y[1000];
  for (i_nom_1 = 1; i_nom_1 <= 1000; i_nom_1 += 1) {
    x[1 * i_nom_1 + -1] = (cos(a[1 * i_nom_1 + -1]));
    y[1 * i_nom_1 + -1] = (sin(a[1 * i_nom_1 + -1]));
  }
{
/* Auto-generated code for call to _auto_kernel_1 */
    typedef float _narray_w;
    _narray_w *d_w;
    cudaMalloc((void **) &d_w, sizeof(float ) * 1000);
    cudaMemcpy(d_w, w, sizeof(float ) * 1000, cudaMemcpyHostToDevice);
    typedef float _narray_Q;
    _narray_Q *d_Q;
    cudaMalloc((void **) &d_Q, sizeof(float ) * 1000);
    cudaMemcpy(d_Q, Q, sizeof(float ) * 1000, cudaMemcpyHostToDevice);
    typedef float _narray_x;
    _narray_x *d_x;
    cudaMalloc((void **) &d_x, sizeof(float ) * 1000);
    cudaMemcpy(d_x, x, sizeof(float ) * 1000, cudaMemcpyHostToDevice);
    typedef float _narray_y;
    _narray_y *d_y;
    cudaMalloc((void **) &d_y, sizeof(float ) * 1000);
    cudaMemcpy(d_y, y, sizeof(float ) * 1000, cudaMemcpyHostToDevice);
    int CUDA_GRID_X;
    CUDA_GRID_X = (1000 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
    int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
    int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_1<<<CUDA_gridSize,CUDA_blockSize>>>(d_w, d_Q, d_x, d_y);
    cudaMemcpy(w, d_w, sizeof(float ) * 1000, cudaMemcpyDeviceToHost);
    cudaMemcpy(Q, d_Q, sizeof(float ) * 1000, cudaMemcpyDeviceToHost);
    cudaMemcpy(x, d_x, sizeof(float ) * 1000, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, sizeof(float ) * 1000, cudaMemcpyDeviceToHost);
  }
  double sum = 0;
  for (i_nom_3 = 1; i_nom_3 <= 1000; i_nom_3 += 1) {
    sum += Q[1 * i_nom_3 + -1];
  }
  double result = 2 * 3.14159265358 * sum;
/* Report the result */
  printf("Result: %f\n",result);
  return 0;
}
