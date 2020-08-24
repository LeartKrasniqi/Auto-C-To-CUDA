/* Simple program to obtain transpose of a matrix */
#include <stdio.h>
#define M 1000
#define N 500
int main()
{
	char A[M][N];
	char B[N][M];
	
	int m = M, n = N;

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
