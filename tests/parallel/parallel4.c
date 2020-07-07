int agon()
{
	int a[10][10];
	int b[10][10];
	for(int i = 1; i <=10; i++)
		for(int j = 1; j <= 8; j++)
		{
			a[i+3][j+4] = b[i][j];
			b[i+2][j+3] = a[i][j];
		}

	return 100;
}

