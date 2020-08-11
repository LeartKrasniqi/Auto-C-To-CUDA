/* Simple program to obtain transpose of a matrix */
#include <stdio.h>
int main()
{
	int m = 1000;
	int n = 500;

	char A[m][n];
	char B[n][m];
	
	/* Initialize */
	srand(time(0));
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			A[i][j] = (rand()%('Z'-'A')) + 'A';

	/* Get the transpose */
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			B[j][i] = A[i][j];

	/* Assertion */
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			if(A[i][j] != B[j][i])
			{
				fprintf(stderr, "ERROR\n");
				exit(-1);
			}

	printf("All good b0ss\n");

	return 0;
}
