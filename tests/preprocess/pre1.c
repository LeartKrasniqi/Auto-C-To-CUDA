int main()
{
	int a[100][2];

	int i, j;
       	i = 0;
	while(i < 100)
	{
	       	j = 0;
	    	while(j < 2)
		{
			a[2*i][2*j] = a[2*i+1][2*j+1];
			j++;
		}
		i++;
	}

	return 2;
}	
