#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main()
{
	int a[100000], b[100000], c[100000];
	int i = 0;
	for(i = 0; i < 100000; i++)
	{
		a[i] = sin(i)*sin(i);
		b[i] = cos(i)*cos(i);
	}

	for(i = 0; i < 100000; i++)
		c[i] = a[i] + b[i];

	double sum = 0;
	for(i = 0; i < 100000; i++)
		sum += c[i];

	printf("Final result (Should be close to 1): %f\n", sum);

	return 0;
}
