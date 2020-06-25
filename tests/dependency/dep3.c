int naarma()
{
	int x[5][5];
	for(int i = 0; i < 5; i++)
		for(int j = 2; j < 5; j+=3)
			x[i*3 + 4*j + 4 + 5][4+5+3+2 + ((3+1)*j) + (7*8)*i] = 1;
}
