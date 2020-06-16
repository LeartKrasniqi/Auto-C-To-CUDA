int main()
{
	int x[4][2];
	for(int i = 3; i > -1; i--)
		for(int j = 0; j < 3; ++j)
			x[i+1][j] = i*j;
}
