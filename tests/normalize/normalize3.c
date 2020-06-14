int f()
{
	char arr[10];
	char brr[4];
	for(int i = 5; i < 10; i+=2)
		for(int j = 2; j < 4; j = j+1)
		{
			arr[i] = (char)i;
			brr[j] = (char)j;
		}

	return 11;
}
		
