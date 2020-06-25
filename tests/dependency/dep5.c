typedef int ARMAAN;
ARMAAN armaan(ARMAAN aRmAaN)
{
	ARMAAN a[10][10];
	ARMAAN b[10][10];
	int c[5];

	/* Should be NOT DEPENDENT */
	for(int i = 0; i < 10; i++)
		for(int j = 0; j < 10; j++)
			a[2*i][2*j] = b[2*i + 1][2*j + 1];

	/* Should be NOT DEPENDENT -- May get skipped */
	for(int i = 0; i < 10; i++)
		for(int j = -1; j < 9; j++)
		       a[2*i][2*(j+1)] = a[2*i + 1][2*(j+1) + 1];	

	/* Should be DEPENDENT */
	for(int i = 1; i < 10; i++)
		for(int j = 1; j < 10; j++)
			a[i][j] = a[i+1][j+1];

	/* Should be NOT DEPENDENT */
	for(int k = 0; k < 5; k++)
		for(int l = 0; l < 5; l++)
		{	
			a[2*k][2*l] = a[2*k+1][2*l+1];
			c[2*k] = c[2*l+1];
			
		}
	

	return 7;
}
