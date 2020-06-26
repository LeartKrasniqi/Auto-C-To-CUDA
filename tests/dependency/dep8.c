long long a_r_m_a_a_n()
{
	int a[10];
	
	/* NO DEPENDENCE */
	for(int i = 1; i <=20; i++)
		for(int j = 2; j <=10; j++)
			a[i + 2*j] = a[i+ 2*j - 1];
	
	/* NO DEPENDENCE */
	for(int i = 0; i < 100; i++)
		a[i+100] = 2*a[i];

	return 1111;
}
