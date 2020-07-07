
int ecsMinFn(int val1,int val2)
{
  if (val1 < val2) 
    return val1;
   else 
    return val2;
}

int ecsMaxFn(int val1,int val2)
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
    for (int ecs_serial_index = 1; ecs_serial_index <= 3; ++ecs_serial_index) {
      for (i = 1 + (ecs_serial_index - 1) * 2; i <= ecsMinFn(1 + (ecs_serial_index - 1) * 2 + (2 - 1),10); ++i) 
        for (j = 1 + (ecs_serial_index - 1) * 3; j <= 8; ++j) {
          a[1 * i + 3][1 * j + 4] = b[1 * i + 0][1 * j + 0];
          b[1 * i + 2][1 * j + 3] = a[1 * i + 0][1 * j + 0];
        }
      for (j = 1 + (ecs_serial_index - 1) * 3; j <= ecsMinFn(1 + (ecs_serial_index - 1) * 3 + (3 - 1),8); ++j) 
        for (i = 1 + (ecs_serial_index - 1) * 2 + 2; i <= 10; ++i) {
          a[1 * i + 3][1 * j + 4] = b[1 * i + 0][1 * j + 0];
          b[1 * i + 2][1 * j + 3] = a[1 * i + 0][1 * j + 0];
        }
    }
  }
  return 100;
}
