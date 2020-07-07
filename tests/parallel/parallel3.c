int main()
{
	int a[10][10];
	int b[10][10];
	int c[10][10];
	int d[10][10];

	for(int i = 0; i < 10; i++)
		for(int j = 0; j < 10; j++)
		{
			a[i+2][j-3] = b[i][j]; //+ a[i][j];
			b[i+3][j-2] = a[i][j];
			c[i][j] = d[i][j];
			a[i][j] = c[i][j]; 
			//c[i][j] = a[i][j] + b[i][j] + c[i][j];
		}

	return 100;

}
