char*** ArMaAn()
{
	int a[10];
	int b[10];
	for(int i = 1; i < 10; i++)
	{
		/* These should be INDEPENDENT */
		a[2*i] = 10;
		a[2*i + 1] = 20;

		/* These should be INDEPENDENT */
		b[2*i] = 30;
		b[2*i + 1] = 40; 
	}

	/* Testing handling of multiple loop nests */
	for(int i = 1; i < 10; i++)
	{
		/* These should be INDEPENDENT */
		//a[2*i] = 100;
		//a[2*i + 1] = 6;
		
		/* These should be DEPENDENT */
		b[2*i] = 10;
		b[i] = 20;
	}

	return 2222;
}	
