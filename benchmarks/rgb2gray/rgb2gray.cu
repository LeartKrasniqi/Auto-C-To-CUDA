/* Convert rgb PNG to grayscale */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./lodepng/lodepng.h"
#define w 512
#define h 512
#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_0(unsigned char image[512][2048],unsigned char gray_image[512][2048])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= 512 && thread_y_id <= 512) {
      unsigned char r = image[1 * thread_x_id + -1][4 * thread_y_id + -4];
      unsigned char g = image[1 * thread_x_id + -1][4 * thread_y_id + -3];
      unsigned char b = image[1 * thread_x_id + -1][4 * thread_y_id + -2];
      unsigned char gray = (unsigned char )(0.21 * ((float )r) + 0.71 * ((float )g) + 0.07 * ((float )b));
      gray_image[1 * thread_x_id + -1][4 * thread_y_id + -4] = gray;
      gray_image[1 * thread_x_id + -1][4 * thread_y_id + -3] = gray;
      gray_image[1 * thread_x_id + -1][4 * thread_y_id + -2] = gray;
      gray_image[1 * thread_x_id + -1][4 * thread_y_id + -1] = ((unsigned char )255);
    }
}

int main(int argc,char **argv)
{
  int col;
  int row;
  if (argc != 2) {
    fprintf(stderr,"Usage: %s [file]\n",argv[0]);
    return - 1;
  }
/* Obtain image */
  char *infile = argv[1];
  int width;
  int height;
  unsigned char *lodeimg;
  unsigned int error = lodepng_decode32_file(&lodeimg,(&width),(&height),infile);
  unsigned char image[512][2048];
  memcpy(image,lodeimg,1048576);
/* Holds output */
  unsigned char gray_image[512][2048];
{
/* Auto-generated code for call to _auto_kernel_0 */
    typedef unsigned char _narray_image[2048];
    _narray_image *d_image;
    cudaMalloc((void **) &d_image, sizeof(unsigned char ) * 512 *(4 * 512));
    cudaMemcpy(d_image, image, sizeof(unsigned char ) * 512 *(4 * 512), cudaMemcpyHostToDevice);
    typedef unsigned char _narray_gray_image[2048];
    _narray_gray_image *d_gray_image;
    cudaMalloc((void **) &d_gray_image, sizeof(unsigned char ) * 512 *(4 * 512));
    cudaMemcpy(d_gray_image, gray_image, sizeof(unsigned char ) * 512 *(4 * 512), cudaMemcpyHostToDevice);
    int CUDA_GRID_X;
    CUDA_GRID_X = (512 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
    int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
    int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(d_image, d_gray_image);
    cudaMemcpy(image, d_image, sizeof(unsigned char ) * 512 *(4 * 512), cudaMemcpyDeviceToHost);
    cudaMemcpy(gray_image, d_gray_image, sizeof(unsigned char ) * 512 *(4 * 512), cudaMemcpyDeviceToHost);
  }
/* Store the output */
  char *outfile = (malloc(strlen("gray_") + strlen(infile) + 1));
  strcpy(outfile,"gray_");
  strcat(outfile,infile);
  error = lodepng_encode32_file(outfile,gray_image,512,512);
  return 0;
}
