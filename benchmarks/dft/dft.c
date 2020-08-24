/* Calculate the Discrete Fourier Transform of a signal */
/* Adapted from: https://batchloaf.wordpress.com/2013/12/07/simple-dft-in-c/ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Assume N is greater than 4 and a power of 2 */
#define N 512
#define PI2 6.2832

int main()
{	
	int i, j, n, k;

	/* Values of sin and cos */
	float sin_vals[N][N], cos_vals[N][N];
	for(i = 0; i < N; i++)
		for(j = 0; j < N; j++)
		{
			sin_vals[i][j] = sin(i*j*PI2 / N);
			cos_vals[i][j] = cos(i*j*PI2 / N);
		}

	
	/* Discrete time signal -- Generate a random signal in range (-1, 1) */
	float x[N];
	srand(time(0));
	for(i = 0; i < N; i++)
		x[i] = ((2.0 * rand()) / RAND_MAX) - 1.0 + sin(PI2 * i * 5.7 / N);

	/* These will hold the DFT x (both real and imaginary parts) */
	float x_re[N/2+1], x_im[N/2+1];

	/* This will hold the power spectrum of x */
	float P[N/2+1];

	/* Calculate DFT and Power Spectrum up to the Nyquist Frequency */
	for(k = 0; k <= N/2; k++)
	{	
		for(n = 0; n < N; n++)
		{
			x_re[k] += x[n]*cos_vals[k][n];//x_re_inter[n];
			x_im[k] -= x[n]*sin_vals[k][k];//x_im_inter[n];
		}

		P[k] = x_re[k]*x_re[k] + x_im[k]*x_im[k];
	}

	return 0;

}

