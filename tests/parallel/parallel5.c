int agon()
{
	int a[10][10];
	int b[10][10];
	int c[10][10];
	int d[10][10];
	for(int i = 1; i <=10; i++)
		for(int j = 1; j <= 8; j++)
		{
			a[i+3][j+4] = b[i][j];
			b[i+2][j+3] = a[i][j];
			c[i+3][j+4] = d[i][j];
			d[i+2][j+3] = c[i][j];
		}

	return 100;
}

