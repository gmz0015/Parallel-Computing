#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acw18mg"		//replace with your user name

#pragma warning(disable : 4996)

void print_help();
int process_command_line(int argc, char *argv[]); // process the arguments
int readFile(); // read header and original pixel values from file
int allocateMemory(); // allocate the two dimension arrays' memory
int runCPU(unsigned short *red_temp, unsigned short *green_temp, unsigned short *blue_temp); // run with CPU mode
int runOPENMP(unsigned short *red_temp, unsigned short *green_temp, unsigned short *blue_temp); // run with OpenMP mode
int writeBinary(); // write header and new pixel values to binary file
int writePlainText(); // write header and new pixel values to plain text file
int freeMemory(); // free memory

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;
typedef enum FORMAT { PPM_BINARY, PPM_PLAIN_TEXT } FORMAT;

/* command line arguments */
unsigned int c = 0; // store c value
char *input_image_name = NULL; // store input image name
char *output_image_name = NULL; // store output image name
MODE execution_mode = OPENMP; // store mode
FORMAT image_format = PPM_BINARY; // store output image format(optional, default is PPM_BINARY)

/* header */
unsigned char magic_number[3]; // store magic number and '\0'
char comment[64] = ""; // store comment
unsigned short width = 0; // store width
unsigned short height = 0; // store height
unsigned short max_color_value = 0; // store maximum color value

/* pixel value */
unsigned short **red; // global red values (two dimension[height][width])
unsigned short **green; // global green values (two dimension[height][width])
unsigned short **blue; // global blue values (two dimension[height][width])

/*
  Parallel Outer Loop
*/
int main(int argc, char *argv[]) {
	unsigned short red_average = 0;
	unsigned short blue_average = 0;
	unsigned short green_average = 0;

	clock_t begin, end;
	double begin_openmp, end_openmp;


	if (process_command_line(argc, argv) == FAILURE) {
		//return 1;

		//Test (Remove when finished)
		printf("arguments wrong!\n");
		//return 0;
	}



	//TODO: read input image file (either binary or plain text PPM) 
	readFile();

	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		//TODO: starting timing here
		begin = clock();

		//TODO: calculate the average colour value
		runCPU(&red_average, &green_average, &blue_average);

		// Output the average colour value for the image
		printf("CPU Average image colour red = %hu, green = %hu, blue = %hu \n", red_average, green_average, blue_average);

		//TODO: end timing here
		end = clock();
		printf("CPU mode execution time took %.0f s and %.0f ms\n", (end - begin) / (float)CLOCKS_PER_SEC, ((end - begin) / (float)CLOCKS_PER_SEC)*1000.0);

		//save the output image file (from last executed mode)
		switch (image_format) {
		case (PPM_BINARY): {
			writeBinary();
			break;
		}
		case (PPM_PLAIN_TEXT): {
			writePlainText();
			break;
		}
		}

		break;
	}
	case (OPENMP): {
		//TODO: starting timing here
		begin_openmp = omp_get_wtime();

		//TODO: calculate the average colour value
		runOPENMP(&red_average, &green_average, &blue_average);

		// Output the average colour value for the image
		printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", red_average, green_average, blue_average);

		//TODO: end timing here
		end_openmp = omp_get_wtime();
		printf("OPENMP mode execution time took %.0f s and %.0f ms\n", (end_openmp - begin_openmp), (end_openmp - begin_openmp)*1000.0);

		//save the output image file (from last executed mode)
		switch (image_format) {
		case (PPM_BINARY): {
			writeBinary();
			break;
		}
		case (PPM_PLAIN_TEXT): {
			writePlainText();
			break;
		}
		}

		break;
	}
	case (CUDA): {
		printf("CUDA Implementation not required for assignment part 1\n");
		break;
	}
	case (ALL): {
		// starting timing here
		begin = clock();

		// calculate the average colour value
		runCPU(&red_average, &green_average, &blue_average);

		// Output the average colour value for the image
		printf("CPU Average image colour red = %d, green = %d, blue = %d \n", red_average, green_average, blue_average);

		// end timing here
		end = clock();
		printf("CPU mode execution time took %.0f s and %.0f ms\n", (end - begin) / (float)CLOCKS_PER_SEC, ((end - begin) / (float)CLOCKS_PER_SEC)*1000.0);

		//save the output image file (from last executed mode)
		switch (image_format) {
		case (PPM_BINARY): {
			writeBinary();
			break;
		}
		case (PPM_PLAIN_TEXT): {
			writePlainText();
			break;
		}
		}



		// read the file again for openmp
		readFile();

		// starting timing here
		begin_openmp = omp_get_wtime();

		// calculate the average colour value
		runOPENMP(&red_average, &green_average, &blue_average);

		// Output the average colour value for the image
		printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", red_average, green_average, blue_average);

		//end timing here
		end_openmp = omp_get_wtime();
		printf("OPENMP mode execution time took %.0f s and %.0f ms\n", (end_openmp - begin_openmp), (end_openmp - begin_openmp)*1000.0);

		//save the output image file (from last executed mode)
		switch (image_format) {
		case (PPM_BINARY): {
			writeBinary();
			break;
		}
		case (PPM_PLAIN_TEXT): {
			writePlainText();
			break;
		}
		}

		break;
	}
	}

	freeMemory();
	getchar();
	return 0;
}

void print_help() {
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name

	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);

	//TODO: read in the mode
	if (!strcmp(argv[2], "CPU"))
		execution_mode = CPU;
	else if (!strcmp(argv[2], "OPENMP"))
		execution_mode = OPENMP;
	else if (!strcmp(argv[2], "CUDA"))
		execution_mode = CUDA;
	else if (!strcmp(argv[2], "ALL"))
		execution_mode = ALL;
	else
		fprintf(stderr, "Error: Wrong mode argument. Correct usage is CPU, OPENMP, CUDA or ALL.\n");

	//TODO: read in the input image name
	input_image_name = argv[4];

	//TODO: read in the output image name
	output_image_name = argv[6];

	//TODO: read in any optional part 3 arguments
	if (argc == 9)
		if (!strcmp(argv[8], "PPM_BINARY"))
			image_format = PPM_BINARY;
		else if (!strcmp(argv[8], "PPM_PLAIN_TEXT"))
			image_format = PPM_PLAIN_TEXT;
		else
			fprintf(stderr, "Error: Wrong image format argument. Correct usage is PPM_BINARY or PPM_PLAIN_TEXT.\n");

	return SUCCESS;
}

int readFile() {
	/* cache */
	char temp[2]; // temporarily store each read result 
	unsigned char *temp_value = (char*)malloc(sizeof(char) * 4); // temporarily store the reading results

	FILE *f = NULL;

	// open file as binary
	f = fopen(input_image_name, "rb");

	if (f == NULL) {
		fprintf(stderr, "Could not open file\n");
	}


	// Read magic number
	fread(magic_number, 3, 1, f);
	magic_number[2] = '\0';
	if (strncmp(magic_number, "P3", 2) == 0) {
		// Plain Text
		printf("Read plain text file is beginning\n");

		for (int i = 0; i < 3; i++) {
			fgets(comment, 64, f);

			// skip the comment part
			if (strncmp(comment, "#", 1) == 0)
				i--;
			else if (width == 0)
				width = (unsigned short)atoi(comment);
			else if (height == 0)
				height = (unsigned short)atoi(comment);
			else
				max_color_value = (unsigned short)atoi(comment);

		}

		// allocate memory for arrays
		allocateMemory();

		// assign value for arrays
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				fscanf(f, "%hu %hu %hu", &red[i][j], &green[i][j], &blue[i][j]);
				//printf("red is: %hu, %hu, %hu\n", red[i][j], green[i][j], blue[i][j]);
			}
		}
		printf("Read plain text file is finished\n");
	}
	else {
		// Binary File
		printf("Read binary file is beginning\n");
		int i = 0;
		int j = 0;
		int k = 0;// iterate binary file without line break


		// read in width
		while (1) {
			fread(temp, 1, 1, f);
			if (!strcmp("\n", temp))
				break;
			else
				temp_value[i++] = temp[0];
		}
		temp_value[i] = '\0';
		width = atoi(temp_value);


		// read in height
		i = 0;
		while (1) {
			fread(temp, 1, 1, f);
			if (!strcmp("\n", temp))
				break;
			else
				temp_value[i++] = temp[0];
		}
		temp_value[i] = '\0';
		height = atoi(temp_value);


		// read in max color value
		i = 0;
		while (1) {
			fread(temp, 1, 1, f);
			if (!strcmp("\n", temp))
				break;
			else
				temp_value[i++] = temp[0];
		}
		temp_value[i] = '\0';
		max_color_value = atoi(temp_value);

		// allocate memory for arrays
		allocateMemory();

		unsigned char *color_temp = (char*)malloc(sizeof(char) * 3 * width * height);
		fread(color_temp, 3, width * height, f);

		// read in red, green and blue into 3 arrays.
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				red[i][j] = *(color_temp + k * 3);
				green[i][j] = *(color_temp + k * 3 + 1);
				blue[i][j] = *(color_temp + k * 3 + 2);
				k++;
			}
		}
		printf("Read binary file is finished\n");
	}


	return 1;
}

/*
  dynamically allocate the memory for two dimension array
*/
int allocateMemory() {
	red = (unsigned short **)malloc(sizeof(unsigned short *) * height);
	for (int i = 0; i < height; i++)
		red[i] = (unsigned short *)malloc(sizeof(unsigned short) * width);

	green = (unsigned short **)malloc(sizeof(unsigned short *) * height);
	for (int i = 0; i < height; i++)
		green[i] = (unsigned short *)malloc(sizeof(unsigned short) * width);

	blue = (unsigned short **)malloc(sizeof(unsigned short *) * height);
	for (int i = 0; i < height; i++)
		blue[i] = (unsigned short *)malloc(sizeof(unsigned short) * width);

	return 1;
}

/*
  CPU mode
*/
int runCPU(unsigned short *red_temp, unsigned short *green_temp, unsigned short *blue_temp) {
	/* iteration */
	unsigned short i, j, k, l = 0;

	/* the number and width of cells */
	unsigned short quotient_row = 0; // the number of square cells in a row
	unsigned short remainder_row = 0; // the width of the rest cell in a row (optional)
	unsigned short quotient_column = 0; // the number of square cells in a column
	unsigned short remainder_column = 0; // the height of the rest cell in a column (optional)

	/* the total number of cells in a row and column */
	int cells_per_row; // the number of cells in a row (decide i)
	int cells_per_column; // the number of cells in a column (decide j)

	/* the total number of cells */
	int cells;

	/* the width and height for a cell */
	unsigned short limitation_width = 0; // the width of a cell
	unsigned short limitation_height = 0; // the height of a cell

	/* average value in a cell */
	unsigned long red_average_part = 0;
	unsigned long green_average_part = 0;
	unsigned long blue_average_part = 0;

	/* average value in all */
	unsigned long red_average_all = 0;
	unsigned long green_average_all = 0;
	unsigned long blue_average_all = 0;

	/* calculate the number and width of cells */
	quotient_row = width / c;
	remainder_row = width % c;
	quotient_column = height / c;
	remainder_column = height % c;

	/* calculate the total number of cells */
	if (remainder_row)
		// if image is not multiples of c
		cells_per_row = quotient_row + 1;
	else
		cells_per_row = quotient_row;

	if (remainder_column)
		// if image is not multiples of c
		cells_per_column = quotient_column + 1;
	else
		cells_per_column = quotient_column;

	cells = cells_per_row * cells_per_column;

	for (i = 0; i < cells_per_column; i++) {
		// loop cells in column

		// change limitaion height of the cell 
		if (i == quotient_column)
			limitation_height = remainder_column;
		else
			limitation_height = c;

		for (j = 0; j < cells_per_row; j++) {
			// loop cells in row

			// initial average in a cell
			red_average_part = 0;
			green_average_part = 0;
			blue_average_part = 0;

			// change limitation width of the cell
			if (j == quotient_row)
				limitation_width = remainder_row;
			else
				limitation_width = c;

			// sum all pixel values in a cell
			for (k = 0; k < limitation_height; k++) {
				// loop pixel in cell row
				for (l = 0; l < limitation_width; l++) {
					// loop pixel in cell column

					// sum up pixel values
					red_average_part += red[i*c + k][j*c + l];
					green_average_part += green[i*c + k][j*c + l];
					blue_average_part += blue[i*c + k][j*c + l];
				}
			}

			// save the sum of pixel values in a cell
			red_average_all += red_average_part;
			green_average_all += green_average_part;
			blue_average_all += blue_average_part;

			// calculate average pixel values
			red_average_part /= (limitation_width * limitation_height);
			green_average_part /= (limitation_width * limitation_height);
			blue_average_part /= (limitation_width * limitation_height);

			// assign the pixel value to average
			for (k = 0; k < limitation_height; k++) {
				// loop column in cell
				for (l = 0; l < limitation_width; l++) {
					// loop row in cell

					red[i*c + k][j*c + l] = (unsigned short)red_average_part;
					green[i*c + k][j*c + l] = (unsigned short)green_average_part;
					blue[i*c + k][j*c + l] = (unsigned short)blue_average_part;

				}
			}
		}
	}

	// calculate all average pixel values
	red_average_all /= (width * height);
	green_average_all /= (width * height);
	blue_average_all /= (width * height);

	// return pixel values
	*red_temp = (unsigned short)red_average_all;
	*green_temp = (unsigned short)green_average_all;
	*blue_temp = (unsigned short)blue_average_all;

	return 1;
}

/*
  OPENMP
*/
int runOPENMP(unsigned short *red_temp, unsigned short *green_temp, unsigned short *blue_temp) {
	/* iteration */
	signed short i, j, k, l = 0;

	/* the number and width of cells */
	unsigned short quotient_row = 0; // the number of square cells in a row
	unsigned short remainder_row = 0; // the width of the rest cell in a row (optional)
	unsigned short quotient_column = 0; // the number of square cells in a column
	unsigned short remainder_column = 0; // the height of the rest cell in a column (optional)

	/* the total number of cells in a row and column */
	int cells_per_row; // the number of cells in a row (decide i)
	int cells_per_column; // the number of cells in a column (decide j)

	/* the total number of cells */
	int cells;

	/* the width and height for a cell */
	unsigned short limitation_width = 0; // the width of a cell
	unsigned short limitation_height = 0; // the height of a cell

	/* average value in a cell */
	unsigned long red_average_part = 0;
	unsigned long green_average_part = 0;
	unsigned long blue_average_part = 0;

	/* average value in all */
	unsigned long red_average_all = 0;
	unsigned long green_average_all = 0;
	unsigned long blue_average_all = 0;

	/* calculate the number and width of cells */
	quotient_row = width / c;
	remainder_row = width % c;
	quotient_column = height / c;
	remainder_column = height % c;

	/* calculate the total number of cells */
	if (remainder_row)
		// if image is not multiples of c
		cells_per_row = quotient_row + 1;
	else
		cells_per_row = quotient_row;

	if (remainder_column)
		// if image is not multiples of c
		cells_per_column = quotient_column + 1;
	else
		cells_per_column = quotient_column;

	cells = cells_per_row * cells_per_column;

#pragma omp parallel private (i, j, k, l, red_average_part, green_average_part, blue_average_part) 
	{
#pragma omp for schedule(dynamic)
		for (i = 0; i < cells_per_column; i++) {
			// loop cells in column

			// change limitaion height of the cell  
			if (i == quotient_column)
				limitation_height = remainder_column;
			else
				limitation_height = c;

			for (j = 0; j < cells_per_row; j++) {
				// loop cells in row

				// change limitaion width of the cell 
				if (j == quotient_row)
					limitation_width = remainder_row;
				else
					limitation_width = c;

				// initial average in a cell
				red_average_part = 0;
				green_average_part = 0;
				blue_average_part = 0;

				// sum all pixel in a cell

				for (k = 0; k < limitation_height; k++) {
					// loop row in cell
					for (l = 0; l < limitation_width; l++) {
						// loop column in cell
						red_average_part += red[i*c + k][j*c + l];
						green_average_part += green[i*c + k][j*c + l];
						blue_average_part += blue[i*c + k][j*c + l];
					}
				}// end first inner loop

			    // sum up all cell average values
#pragma omp atomic
				red_average_all += red_average_part;
#pragma omp atomic
				green_average_all += green_average_part;
#pragma omp atomic
				blue_average_all += blue_average_part;

				// calculate average pixel values
				red_average_part /= (limitation_width * limitation_height);
				green_average_part /= (limitation_width * limitation_height);
				blue_average_part /= (limitation_width * limitation_height);

				// change the pixel to average
				for (k = 0; k < limitation_height; k++) {
					// loop column in cell
					for (l = 0; l < limitation_width; l++) {
						// loop row in cell
						red[i*c + k][j*c + l] = (unsigned short)red_average_part;
						green[i*c + k][j*c + l] = (unsigned short)green_average_part;
						blue[i*c + k][j*c + l] = (unsigned short)blue_average_part;

					}
				}// end second inner loop
			}// end for (j)
		}// end for (i)
	}// end parallel

	// calculate all average pixel values
	red_average_all /= width * height;
	green_average_all /= width * height;
	blue_average_all /= width * height;

	// return pixel values
	*red_temp = (unsigned short)red_average_all;
	*green_temp = (unsigned short)green_average_all;
	*blue_temp = (unsigned short)blue_average_all;

	return 1;
}
/*
  write to binary file
*/
int writeBinary() {
	int i, j;
	FILE *f = NULL;

	printf("Start write binary file\n");

	// open file as binary, read
	f = fopen(output_image_name, "wb");

	// write magic number
	fwrite("P6\n", sizeof(char), sizeof(magic_number) / sizeof(char), f);

	// store chache for writing
	char itoa_temp[64] = "";

	// write width
	itoa(width, itoa_temp, 10);
	itoa_temp[strlen(itoa_temp)] = '\n';
	fwrite(itoa_temp, sizeof(char), strlen(itoa_temp), f);

	// write height
	itoa(height, itoa_temp, 10);
	itoa_temp[strlen(itoa_temp)] = '\n';
	fwrite(itoa_temp, sizeof(char), strlen(itoa_temp), f);

	// write max color value
	itoa(max_color_value, itoa_temp, 10);
	itoa_temp[strlen(itoa_temp)] = '\n';
	fwrite(itoa_temp, 1, 4, f);

	// write pixel
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			fwrite(&red[i][j], 1, 1, f);
			fwrite(&green[i][j], 1, 1, f);
			fwrite(&blue[i][j], 1, 1, f);
		}
	}


	fclose(f);
	printf("Write binary file is finished!\n");

	return 1;
}

/*
  write to plain text file
*/
int writePlainText() {
	int i, j;
	FILE *f = NULL;

	printf("Start write plain text file\n");

	// open file as binary, read
	f = fopen(output_image_name, "wb");

	// write header
	fprintf(f, "%s\n%d\n%d\n%d\n", "P3", width, height, max_color_value);

	// write pixel information
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			fprintf(f, "%d %d %d", red[i][j], green[i][j], blue[i][j]);

			// change for end of a line
			if ((j + 1) < width)
				fprintf(f, "\t");
			else
				fprintf(f, "\n");
		}
	}

	fclose(f);
	printf("Write plain text file is finished!\n");

	return 1;
}

/*
  free memory
*/
int freeMemory() {
	int i;

	for (i = 0; i < height; i++)
		free(red[i]);
	free(red);
	for (i = 0; i < height; i++)
		free(green[i]);
	free(green);
	for (i = 0; i < height; i++)
		free(blue[i]);
	free(blue);

	return 1;
}