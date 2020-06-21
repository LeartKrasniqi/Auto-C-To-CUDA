int main()
{
	int a[10];
	int b[3][4];
	int x = 1;
	int p = 3;
	for(int i = 1; i < 15; i++)
		for(int j = 1; j <= x; j++)
		{
			/* Affine */
			/*
			int z = 1;
			a[z*i] = 1;
			a[i] = 1;
			a[i + 1] = 1;
			a[3*i] = 1;
			a[3*i-2] = 1;
			a[i/5] = b[3*j/5][2];
			p++;
			a[x+i] = i*j;
			a[2+3] = a[x+1];
			b[i+j][3*x] = a[j+x];
			a[i+j] = b[3*j + 2*i][z*x + p*j + i];
			*/

			/* Non-affine */
			//x = 4;
			//i += 1;
			//j++;
			//a[i] = a[x*i];
			//a[i*j] = 1;
			//a[i/j] = 5;
			a[i+j] = a[3*j*i];
			//a[a[i]] = 1;

		}
}
