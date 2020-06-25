void armaaaaaaaaan()
{
	int a[2][3][5];
	int b[1][1][1];
	for(int i = 0; i < 5; i++)
		for(int j = 1; j < 5; j+=1)
			for(int k = 200; k > 0; k += -3)
			{
				int x = 0;
				x++;
				a[i][j][k] = 3*i + 4*j + 4*k;
				a[k][j][i] = 4;
				//b[0][j][2] = a[1][1][2] + b[3*i+1][4*j][k+1];
			}
}
