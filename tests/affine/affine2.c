int main()
{
	int a[10];
	for(int i = 1; i < 15; i++)
	{
		a[i] = 1;
		a[i + 1] = 1;
		a[3*i] = 1;
		a[3*i-2] = 1;
	}
}
