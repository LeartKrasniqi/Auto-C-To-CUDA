/* Test file */

int main()
{
	int x = 0;
	x++;
	int arr[4][6] = {0};
	for(int i = 0, z = 1; i < 4; i++)
		for(int j = 0; j < 6; ++j)
			if(i != j)
				arr[i][j] = i+j;
			else
				z++;

	// Comment
	return 0;
}
