/* Convert rgb PNG to grayscale */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./lodepng/lodepng.h"
#define w 512
#define h 512

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		fprintf(stderr, "Usage: %s [file]\n", argv[0]);
		return -1;
	}

	/* Obtain image */
	char *infile = argv[1];
	int width, height;
	unsigned char *lodeimg;
	unsigned error = lodepng_decode32_file(&lodeimg, &width, &height, infile);
	unsigned char image[h][4*w];
	memcpy(image, lodeimg, 4*w*h);

	/* Holds output */
	unsigned char gray_image[h][4*w];

	/* Perform the conversion */
	for(int row = 0; row < h; row++)
		for(int col = 0; col < 4*w; col += 4)
		{
			unsigned char r = image[row][col];
			unsigned char g = image[row][col+1];
			unsigned char b = image[row][col+2];
			unsigned char gray = 0.21f*r + 0.71f*g + 0.07f*b;

			gray_image[row][col] = gray;
			gray_image[row][col+1] = gray;
			gray_image[row][col+2] = gray;
			gray_image[row][col+3] = 255;		
		}

	
	/* Store the output */
	char *outfile = malloc( strlen("gray_") + strlen(infile) + 1 );
	strcpy(outfile, "gray_");
	strcat(outfile, infile);
	error = lodepng_encode32_file(outfile, gray_image, w, h);

	return 0;
}
