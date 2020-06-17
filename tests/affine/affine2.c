int main()
{
	int a[10];
	int x = 1;
	int p = 3;
	for(int i = 1; i < 15; i++)
		for(int j = 0; j < 10; j++)
		{
			/* Affine */
			a[i] = 1;
			a[i + 1] = 1;
			a[3*i] = 1;
			a[3*i-2] = 1;
			x++;
			a[1+i] = i*j;
			a[2+3] = a[x+1];
						
			/* Non-affine */
			a[i] = a[x*i];
			a[i+j] = a[3*j+ 2*i];	
			a[i++] = 1;
			a[i*j] = 1;
			a[i+j] = a[3*j*i];
			a[a[i]] = 1;

		}
}
