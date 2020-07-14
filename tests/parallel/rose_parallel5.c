
int ecsMinFn(int val1,int val2)
{
  if (val1 > val2) 
    return val2;
   else 
    return val1;
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
  int c[10][10];
  int d[10][10];
{
    for (int ecs_serial_index_0 = 1; ecs_serial_index_0 <= 3; ++ecs_serial_index_0) {
      for (i = 1 + (ecs_serial_index_0 - 1) * 2; i <= ecsMinFn(1 + (ecs_serial_index_0 - 1) * 2 + (2 - 1),10); ++i) 
        for (j = 1 + (ecs_serial_index_0 - 1) * 3; j <= 8; ++j) {
          c[1 * i + 3][1 * j + 4] = d[1 * i + 0][1 * j + 0];
          d[1 * i + 2][1 * j + 3] = c[1 * i + 0][1 * j + 0];
        }
      for (j = 1 + (ecs_serial_index_0 - 1) * 3; j <= ecsMinFn(1 + (ecs_serial_index_0 - 1) * 3 + (3 - 1),8); ++j) 
        for (i = 1 + (ecs_serial_index_0 - 1) * 2 + 2; i <= 10; ++i) {
          c[1 * i + 3][1 * j + 4] = d[1 * i + 0][1 * j + 0];
          d[1 * i + 2][1 * j + 3] = c[1 * i + 0][1 * j + 0];
        }
    }
    for (int ecs_serial_index_1 = 1; ecs_serial_index_1 <= 3; ++ecs_serial_index_1) {
      for (i = 1 + (ecs_serial_index_1 - 1) * 2; i <= ecsMinFn(1 + (ecs_serial_index_1 - 1) * 2 + (2 - 1),10); ++i) 
        for (j = 1 + (ecs_serial_index_1 - 1) * 3; j <= 8; ++j) {
          a[1 * i + 3][1 * j + 4] = b[1 * i + 0][1 * j + 0];
          b[1 * i + 2][1 * j + 3] = a[1 * i + 0][1 * j + 0];
        }
      for (j = 1 + (ecs_serial_index_1 - 1) * 3; j <= ecsMinFn(1 + (ecs_serial_index_1 - 1) * 3 + (3 - 1),8); ++j) 
        for (i = 1 + (ecs_serial_index_1 - 1) * 2 + 2; i <= 10; ++i) {
          a[1 * i + 3][1 * j + 4] = b[1 * i + 0][1 * j + 0];
          b[1 * i + 2][1 * j + 3] = a[1 * i + 0][1 * j + 0];
        }
    }
  }
  return 100;
}
