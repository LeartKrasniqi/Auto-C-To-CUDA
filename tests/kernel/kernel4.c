int main()
{
	int n = 10;
	char x[n], y[n], z[n];
	for(int p = 0; p < n; p++)
	{
		y[p] = z[p];
		x[p] = y[p];
	} 
/*
	for(int p = 0; p < n; p++)
		y[p] = z[p];
	
	for(int p = 0; p < n; p++)
		x[p] = y[p];
*/
	return 0;
}
