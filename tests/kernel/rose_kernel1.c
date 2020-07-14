__global__ 

void _auto_kernel_0(int a[10])
{
  int thread_x_id;thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_x_id) 
    if (thread_x_id <= 10) {
      a[2 * thread_x_id + -2] = a[2 * thread_x_id + -1];
    }
}

int main()
{
  int i;
  int a[10];
  for (i = 1; i <= 10; i += 1) {
    a[2 * i + -2] = a[2 * i + -1];
  }
  return 1;
}
