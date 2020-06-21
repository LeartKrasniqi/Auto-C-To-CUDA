void armaan()
{
	int a[2][2][2];
	int b[2][2][2];

	for(int i = 16; i > 5; i--)
		for(int j = 2; j < 10; j += 2)
			for(int k = 0; k < 2; k++)
			{
				int dummy = 7;
				a[i][2*k][3*j + 2] = b[3*i +2*j+7*k+1][1][i+j] + dummy;
				dummy++;
					
			}

}	
