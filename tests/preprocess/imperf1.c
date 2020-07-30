int main()
{
	int a[5][5], b[5][5];
#if 0
	/* This should be convertible */
	for(int i = 0; i < 5; i++)
	{
		a[i][i] = 1;
		b[i][0] = 1;
		for(int j = 0; j < 5; j++)
		{
			int x = a[i][j];
			x = b[j][i];
		}
	}
#endif
	/* This should NOT be convertible */
	for(int i = 0; i < 5; i++)
	{
		a[i][i] = a[i][i];
		for(int j = 0; j < 5; j++)
			a[i][j] = a[i][j];
	}

	return 0;
}
