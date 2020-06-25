void ************armaaaaaaaaaaaaan()
{
	int a[10][10][10];
	int b[1];

	/* NOT DEPENDENT */
	for(int i = 0; i < 10; i++)
		for(int j = 0; j < 10; j++)
			a[0][0][0] = a[1][0][0];

	/* NOT DEPENDENT */
	for(int i = 0; i < 10; i++)
		for(int j = 0; j < 10; j++)
			a[0][i][2*j] = a[0][i][2*j+1];

	/* DEPENDENT */
	for(int i = 0; i < 10; i++)
		for(int j = 0; j < 10; j++)
			a[1][2][0] = a[1][2][i];

	/* DEPDENDENT */
	for(int k = 0; k < 10; k++)
		b[0] = b[k];
	
	/* NOT DEPENDENT */
	for(int x = 0; x < 10; x++)
		for(int y = 0; y < 10; y++)
			for(int z = 0; z < 10; z++)
				a[0][0][2*x + 2*y + 2*z] = a[0][0][2*x+1 + 2*y+1 + 2*z+1];

}
