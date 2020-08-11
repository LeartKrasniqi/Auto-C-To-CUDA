/* Convert rgb PNG to grayscale */
//#include "./lodepng.h"

int main()
{
	/* Obtain image */
	char *infile = "test.png";
	int w, h;
	unsigned char *lodeimg;
	unsigned error = lodepng_decode32_file(&lodeimg, &w, &h, infile);
	unsigned char image[h][4*w];
	memcpy(image, lodeimg, 4*w*h);

	/* Holds output */
	float gray_image[h][w];

	/* Perform the conversion */
	for(int row = 0; row < h; row++)
		for(int col = 0; col < w; col += 4)
		{
			unsigned char r = image[row][col];
			unsigned char g = image[row][col+1];
			unsigned char b = image[row][col+2];

			gray_image[row][col] = 0.21f*r + 0.71f*g + 0.07f*b;
		}

	
	/* Store the output */
	char *outfile = "test_out.png";
	error = lodepng_encode32_file(outfile, gray_image, w, h);

	return 0;
}
