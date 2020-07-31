int main()
{
	int a[5][5], b[5][5], y;

	/* This should be convertible */
	for(int i = 0; i < 5; i++)
	{
		a[i][i] = 1;
		b[i][0] = 1;
		for(int j = 0; j < 5; j++)
			b[i][j] = a[i][j];
	}

	/* This should NOT be convertible */
	for(int i = 0; i < 5; i++)
	{
		a[i][i] = b[i][i];
		for(int j = 0; j < 5; j++)
			b[i][j] = a[i][j];
	}

	return 0;
}
