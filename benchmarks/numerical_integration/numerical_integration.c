/* Benchmark that calculate the integral of F(x) over the interval [A,B] */
#include <stdio.h>
#define NUM_INTERVALS 1000000
#define F(x) (x)*(x)
#define A 0
#define B 10

int main()
{
	/* Array to hold answer */
	float arr[NUM_INTERVALS];

	/* Step size */
	float delta = (float)(B-A)/(float)NUM_INTERVALS;

	/* Perform integration */
	for(int i = 0; i < NUM_INTERVALS; i++)
	{
		float x = A + (float)i * delta;
		arr[i] = F(x) + F(x*delta);
	}

	/* Add up the heights of the rectangles */
	double sum = 0;
	for(int i = 0; i < NUM_INTERVALS; i++)
		sum += arr[i];

	/* Multiply by width of rectangle */
	sum *= delta/2.0;

	/* Report result */
	printf("Result: %f\n", sum);

	return 0;
}

