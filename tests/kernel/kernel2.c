int main()
{
	int a[10][3];
	for(int i = 0; i < 10; i++)
		a[2*i][2] = a[2*i + 1][2];

	return 1;
}
