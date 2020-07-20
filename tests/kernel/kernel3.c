int main()
{
	int a[10][20];
	for(int i = 0; i < 10; i++)
		for(int j = 0; j < 20; j++)
			a[2*i][2*j+1] = a[2*i + 1][2*j];

	return 1;
}
