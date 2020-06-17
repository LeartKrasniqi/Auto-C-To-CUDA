int main()
{
	int a[10][10];
	int b[11][10];
	for(int i = 1; i < 9; i += 2)
	       for(int j = 0; j < 5; j++)
		       //a[2*i][j+2] = b[2*j + 1][i];
	       {a[2*i][j+2] = b[3*j + 5][7*i] + b[9*i][i+1];}	
}
