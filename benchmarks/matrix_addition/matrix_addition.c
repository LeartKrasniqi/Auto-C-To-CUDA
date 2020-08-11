/* Simple benchmark to test matrix addition */
#include <stdio.h>
#include <math.h>

#define ROWS 100000
#define COLS 100000

int main()
{
	/* Declare three arrays: C = A + B */
	float A[ROWS][COLS], B[ROWS][COLS], C[ROWS][COLS];

	/* Initialize */
	for(int i = 0; i < ROWS; i++)
		for(int j = 0; j < COLS; j++)
		{
			A[i][j] = sin(i+j)*sin(i+j);
			B[i][j] = cos(i+j)*cos(i+j);
		}

	/* Perform addition */
	for(int i = 0; i < ROWS; i++)
		for(int j = 0; j < COLS; j++)
			C[i][j] = A[i][j] + B[i][j];


	/* Check the result */
	double sum = 0;
	for(int i = 0; i < ROWS; i++)
		for(int j = 0; j < COLS; j++)
			sum += C[i][j];

	/* Report the result */
	double r = (double)ROWS;
	double c = (double)COLS;
	printf("Result (Should be close to 1.00) : %f\n", sum/(r*c));

	return 0;
}

