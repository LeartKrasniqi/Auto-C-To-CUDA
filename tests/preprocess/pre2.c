int main()
{
	int a[100][2][4];

	int i, j, k;
       	i = 0;
	while(i < 100)
	{
	       	j = 0;
	    	while(j < 2)
		{
			k = 15;
			while(k)
			{
				a[2*i][2*j][k] = a[2*i+1][2*j+1][k];
				k--;
			}
			j++;
		}
		i++;
	}

	return 2;
}	
