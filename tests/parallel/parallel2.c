int main()
{
	int a[10], b[10], c[10], d[10];
	for(int i = 0; i < 10; i++)
	{
		a[2*i] = d[i+1];
		b[i+1] = c[i];
		c[i] = a[i+1];

	}
}
