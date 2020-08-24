/* Calculate the Discrete Fourier Transform of a signal */
/* Adapted from: https://batchloaf.wordpress.com/2013/12/07/simple-dft-in-c/ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* Assume N is greater than 4 and a power of 2 */
#define N 512
#define PI2 6.2832
#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_2(float x_re[257],float x_im[257],float P[257])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= 257) {
      P[1 * thread_x_id + -1] = x_re[1 * thread_x_id + -1] * x_re[1 * thread_x_id + -1] + x_im[1 * thread_x_id + -1] * x_im[1 * thread_x_id + -1];
    }
}

__global__ void _auto_kernel_1(float sin_vals[512][512],float x[512],float x_im[257])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= 257 && thread_y_id <= 512) {
//x_im_inter[n];
      x_im[1 * thread_x_id + -1] -= x[1 * thread_y_id + -1] * sin_vals[1 * thread_x_id + -1][1 * thread_x_id + -1];
    }
}

__global__ void _auto_kernel_0(float cos_vals[512][512],float x[512],float x_re[257])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= 257 && thread_y_id <= 512) {
//x_re_inter[n];
      x_re[1 * thread_x_id + -1] += x[1 * thread_y_id + -1] * cos_vals[1 * thread_x_id + -1][1 * thread_y_id + -1];
    }
}

int main()
{
  int i;
  int j;
  int n;
  int k;
/* Values of sin and cos */
  float sin_vals[512][512];
  float cos_vals[512][512];
  for (i = 1; i <= 512; i += 1) {
    for (j = 1; j <= 512; j += 1) {
      sin_vals[1 * i + -1][1 * j + -1] = (sin(((1 * i + -1) * (1 * j + -1)) * 6.2832 / 512));
      cos_vals[1 * i + -1][1 * j + -1] = (cos(((1 * i + -1) * (1 * j + -1)) * 6.2832 / 512));
    }
  }
/* Discrete time signal -- Generate a random signal in range (-1, 1) */
  float x[512];
  srand((time(0)));
  for (i = 1; i <= 512; i += 1) {
    x[1 * i + -1] = (2.0 * (rand()) / 2147483647 - 1.0 + sin(6.2832 * (1 * i + -1) * 5.7 / 512));
  }
/* These will hold the DFT x (both real and imaginary parts) */
  float x_re[257];
  float x_im[257];
/* This will hold the power spectrum of x */
  float P[257];
{
{
{
/* Auto-generated code for call to _auto_kernel_0 */
        typedef float _narray_cos_vals[512];
        _narray_cos_vals *d_cos_vals;
    cudaMalloc((void **) &d_cos_vals, sizeof(float ) * 512 * 512);
    cudaMemcpy(d_cos_vals, cos_vals, sizeof(float ) * 512 * 512, cudaMemcpyHostToDevice);
        typedef float _narray_x;
        _narray_x *d_x;
    cudaMalloc((void **) &d_x, sizeof(float ) * 512);
    cudaMemcpy(d_x, x, sizeof(float ) * 512, cudaMemcpyHostToDevice);
        typedef float _narray_x_re;
        _narray_x_re *d_x_re;
    cudaMalloc((void **) &d_x_re, sizeof(float ) *(256 + 1));
    cudaMemcpy(d_x_re, x_re, sizeof(float ) *(256 + 1), cudaMemcpyHostToDevice);
        int CUDA_GRID_X;
    CUDA_GRID_X = (512 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
        int CUDA_GRID_Y;
    CUDA_GRID_Y = (512 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
        int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(d_cos_vals, d_x, d_x_re);
    cudaMemcpy(cos_vals, d_cos_vals, sizeof(float ) * 512 * 512, cudaMemcpyDeviceToHost);
    cudaMemcpy(x, d_x, sizeof(float ) * 512, cudaMemcpyDeviceToHost);
    cudaMemcpy(x_re, d_x_re, sizeof(float ) *(256 + 1), cudaMemcpyDeviceToHost);
      }
    }
{
{
/* Auto-generated code for call to _auto_kernel_1 */
        typedef float _narray_sin_vals[512];
        _narray_sin_vals *d_sin_vals;
    cudaMalloc((void **) &d_sin_vals, sizeof(float ) * 512 * 512);
    cudaMemcpy(d_sin_vals, sin_vals, sizeof(float ) * 512 * 512, cudaMemcpyHostToDevice);
        typedef float _narray_x;
        _narray_x *d_x;
    cudaMalloc((void **) &d_x, sizeof(float ) * 512);
    cudaMemcpy(d_x, x, sizeof(float ) * 512, cudaMemcpyHostToDevice);
        typedef float _narray_x_im;
        _narray_x_im *d_x_im;
    cudaMalloc((void **) &d_x_im, sizeof(float ) *(256 + 1));
    cudaMemcpy(d_x_im, x_im, sizeof(float ) *(256 + 1), cudaMemcpyHostToDevice);
        int CUDA_GRID_X;
    CUDA_GRID_X = (512 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
        int CUDA_GRID_Y;
    CUDA_GRID_Y = (512 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
        int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_1<<<CUDA_gridSize,CUDA_blockSize>>>(d_sin_vals, d_x, d_x_im);
    cudaMemcpy(sin_vals, d_sin_vals, sizeof(float ) * 512 * 512, cudaMemcpyDeviceToHost);
    cudaMemcpy(x, d_x, sizeof(float ) * 512, cudaMemcpyDeviceToHost);
    cudaMemcpy(x_im, d_x_im, sizeof(float ) *(256 + 1), cudaMemcpyDeviceToHost);
      }
    }
{
/* Auto-generated code for call to _auto_kernel_2 */
      typedef float _narray_x_re;
      _narray_x_re *d_x_re;
    cudaMalloc((void **) &d_x_re, sizeof(float ) *(256 + 1));
    cudaMemcpy(d_x_re, x_re, sizeof(float ) *(256 + 1), cudaMemcpyHostToDevice);
      typedef float _narray_x_im;
      _narray_x_im *d_x_im;
    cudaMalloc((void **) &d_x_im, sizeof(float ) *(256 + 1));
    cudaMemcpy(d_x_im, x_im, sizeof(float ) *(256 + 1), cudaMemcpyHostToDevice);
      typedef float _narray_P;
      _narray_P *d_P;
    cudaMalloc((void **) &d_P, sizeof(float ) *(256 + 1));
    cudaMemcpy(d_P, P, sizeof(float ) *(256 + 1), cudaMemcpyHostToDevice);
      int CUDA_GRID_X;
    CUDA_GRID_X = (1 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
      int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
      int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_2<<<CUDA_gridSize,CUDA_blockSize>>>(d_x_re, d_x_im, d_P);
    cudaMemcpy(x_re, d_x_re, sizeof(float ) *(256 + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(x_im, d_x_im, sizeof(float ) *(256 + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(P, d_P, sizeof(float ) *(256 + 1), cudaMemcpyDeviceToHost);
    }
  }
  return 0;
}
