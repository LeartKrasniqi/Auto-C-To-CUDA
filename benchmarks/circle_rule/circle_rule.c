/* Computes quadrature rules (i.e. circumference) for unit circle in 2D */
/* Adapted from: https://people.sc.fsu.edu/~jburkardt/c_src/circle_rule/circle_rule.html */
#include <stdio.h>
#define NUM_ANGLES 100000
#define PI 3.14159265358
#define F(x,y) x*y

int main()
{
	float w[NUM_ANGLES];	/* Weights */
	float a[NUM_ANGLES];	/* Angles */
	float Q[NUM_ANGLES];	/* Result */

	/* Calculate the weights and angles */
	for(int i = 0; i < NUM_ANGLES; i++)
	{
		w[i] = 1.0 / (float)NUM_ANGLES;
		a[i] = 2.0 * PI * (float)i / (float)NUM_ANGLES;
	}

	/* Useful sin/cos values */
       	float x[NUM_ANGLES], y[NUM_ANGLES];
	for(int i = 0; i < NUM_ANGLES; i++)
	{
		x[i] = cos(a[i]);
		y[i] = sin(a[i]);	
	}	
	
	/* Perform the rule */
	for(int i = 0; i < NUM_ANGLES; i++)
		Q[i] = w[i] * F(x[i],y[i]);

	double sum = 0;
	for(int i = 0; i < NUM_ANGLES; i++)
		sum += Q[i];

	double result = 2*PI*sum;

	/* Report the result */
	printf("Result: %f\n", result);

	return 0;
}

