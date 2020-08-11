/* Convert rgb PNG to grayscale */
#include "./lodepng/lodepng.h"
#define CUDA_BLOCK_X 128
#define CUDA_BLOCK_Y 1
#define CUDA_BLOCK_Z 1

__global__ void _auto_kernel_0(int w,int h,unsigned char image[h][4 * w],unsigned char gray_image[h][4 * w])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_id;thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_x_id && thread_y_id) 
    if (thread_x_id <= (h + 0) / 1 && thread_y_id <= (4 * w + 3) / 4) {
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

int main()
{
  int col;
  int row;
/* Obtain image */
  char *infile = "test.png";
  int w;
  int h;
  unsigned char *lodeimg;
  unsigned int error = lodepng_decode32_file(&lodeimg,(&w),(&h),infile);
  unsigned char image[h][4 * w];
  memcpy(image,lodeimg,(4 * w * h));
/* Holds output */
  unsigned char gray_image[h][4 * w];
{
/* Auto-generated code for call to _auto_kernel_0 */
    typedef unsigned char _narray_image[4 * w];
    _narray_image *d_image;
    cudaMalloc((void **) &d_image, sizeof(unsigned char ) * h *(4 * w));
    cudaMemcpy(d_image, image, sizeof(unsigned char ) * h *(4 * w), cudaMemcpyHostToDevice);
    typedef unsigned char _narray_gray_image[4 * w];
    _narray_gray_image *d_gray_image;
    cudaMalloc((void **) &d_gray_image, sizeof(unsigned char ) * h *(4 * w));
    cudaMemcpy(d_gray_image, gray_image, sizeof(unsigned char ) * h *(4 * w), cudaMemcpyHostToDevice);
    int CUDA_GRID_X;
    CUDA_GRID_X = (1 + CUDA_BLOCK_X - 1)/CUDA_BLOCK_X;
    int CUDA_GRID_Y;
    CUDA_GRID_Y = (1 + CUDA_BLOCK_Y - 1)/CUDA_BLOCK_Y;
    int CUDA_GRID_Z;
    CUDA_GRID_Z = (1 + CUDA_BLOCK_Z - 1)/CUDA_BLOCK_Z;
    const dim3 CUDA_blockSize(CUDA_BLOCK_X, CUDA_BLOCK_Y, CUDA_BLOCK_Z);
    const dim3 CUDA_gridSize(CUDA_GRID_X, CUDA_GRID_Y, CUDA_GRID_Z);
    _auto_kernel_0<<<CUDA_gridSize,CUDA_blockSize>>>(w, h, d_image, d_gray_image);
    cudaMemcpy(image, d_image, sizeof(unsigned char ) * h *(4 * w), cudaMemcpyDeviceToHost);
    cudaMemcpy(gray_image, d_gray_image, sizeof(unsigned char ) * h *(4 * w), cudaMemcpyDeviceToHost);
  }
/* Store the output */
  char *outfile = "test_out.png";
  error = lodepng_encode32_file(outfile,gray_image,w,h);
  return 0;
}
