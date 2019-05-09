#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acw18mg"

#pragma warning(disable : 4996)

void print_help();

// process the arguments
int process_command_line(int argc, char *argv[]); 

// read header and original pixel values from file
int readFile(); 

/* Allocate Memory */
int allocateMemory(); 

int allocateCUDAMemory(unsigned short **d_red_input, unsigned short **d_red_output, unsigned short **d_green_input, unsigned short **d_green_output, unsigned short **d_blue_input, unsigned short **d_blue_output);

void checkCUDAError(const char*);

/* Run */
int runCPU(); 
int runOPENMP(); 
void runCUDA();

void vec2matrix();

/* Write File */
int writeBinary(); 
int writePlainText(); 

/* Free Memory */
int freeMemory(); 

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;
typedef enum FORMAT { PPM_BINARY, PPM_PLAIN_TEXT } FORMAT;

/* command line arguments */
unsigned int c = 0; // store c value
char *input_image_name = NULL; // store input image name
char *output_image_name = NULL; // store output image name
MODE execution_mode = OPENMP; // store mode
FORMAT image_format = PPM_BINARY; // store output image format(optional, default is PPM_BINARY)

/* header */
char magic_number[3]; // store magic number and '\0'
char comment[64] = ""; // store comment
unsigned short width = 0; // store width
unsigned short height = 0; // store height
unsigned short max_color_value = 0; // store maximum color value

/* pixel value - Vector */
unsigned short *red_vector; // global red values (two dimension[height][width])
unsigned short *green_vector; // global green values (two dimension[height][width])
unsigned short *blue_vector; // global blue values (two dimension[height][width])

/* pixel value - Matrix */
unsigned short **red; // global red values (two dimension[height][width])
unsigned short **green; // global green values (two dimension[height][width])
unsigned short **blue; // global blue values (two dimension[height][width])

/*
  Parallel Outer Loop
*/
int main(int argc, char *argv[]) {
	double begin_openmp, end_openmp;

	if (process_command_line(argc, argv) == FAILURE) {
		return 0;
	}

	// Read input image file (either binary or plain text PPM) 
	readFile();

	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		// Calculate the average colour value
		runCPU();

		// Save the output image file (from last executed mode)
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
		// Calculate the average colour value
		runOPENMP();

		// Save the output image file (from last executed mode)
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
		runCUDA();
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
	case (ALL): {
		/* CPU */
		// Calculate the average colour value
		runCPU();

		// Save the output image file (from last executed mode)
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


		/* OPENMP */
		readFile();
		// calculate the average colour value
		runOPENMP();

		// Save the output image file (from last executed mode)
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

		/* CUDA */
		readFile();
		runCUDA();

		// Save the output image file (from last executed mode)
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

/*
  CUDA
*/

__device__ unsigned long computeCell(unsigned int c, int start_point_row, int start_point_column, unsigned short **d_color) {
	unsigned long average_return;
	double average_pixel = 0;
	int i, j;
	for (i = 0; i < c; i++) {
		for (j = 0; j < c; j++) {
			average_pixel += d_color[start_point_row * c + i][start_point_column * c + j];
		}
	}
	average_return = average_pixel;
	printf("%d-%d: %.0f\n", start_point_row, start_point_column, average_pixel);
	average_pixel /= (c*c);
	for (i = 0; i < c; i++) {
		for (j = 0; j < c; j++) {
			d_color[start_point_row * c + i][start_point_column * c + j] = (unsigned short) average_pixel;
		}
	}
	return average_return;
}

__device__ unsigned long *red_average;
__device__ unsigned long *green_average;
__device__ unsigned long *blue_average;

__global__ void assignCell(unsigned int *c, unsigned short **d_red, unsigned short **d_green, unsigned short **d_blue)
{
	// threadIdx.x --- row
	// threadIdx.y --- column
	
	printf("%d-%d: %d-%d-%d\n", threadIdx.x, threadIdx.y, *red_average, *green_average, *blue_average);

	*red_average += computeCell(*c, threadIdx.x, threadIdx.y, d_red);
	*green_average += computeCell(*c, threadIdx.x, threadIdx.y, d_green);
	*blue_average += computeCell(*c, threadIdx.x, threadIdx.y, d_blue);

	printf("%d-%d: %d-%d-%d\n", threadIdx.x, threadIdx.y, *red_average, *green_average, *blue_average);
}


void runCUDA()
{
	printf("\n=============== Start Run CUDA! ===============\n");
	/* Set Clock */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// TODO remember to change
	int size = (width / c) * (height / c);
	unsigned int *d_c;
	// Red
	unsigned short **h_red = (unsigned short **)malloc(sizeof(unsigned short *) * height);
	unsigned short **d_red;
	unsigned short *d_red_data;
	//Green
	unsigned short **h_green = (unsigned short **)malloc(sizeof(unsigned short *) * height);
	unsigned short **d_green;
	unsigned short *d_green_data;
	// Blue
	unsigned short **h_blue = (unsigned short **)malloc(sizeof(unsigned short *) * height);
	unsigned short **d_blue;
	unsigned short *d_blue_data;

	unsigned long h_red_average = 0;
	//unsigned long *red_average;
	unsigned long h_green_average = 0;
	//unsigned long *green_average;
	unsigned long h_blue_average = 0;
	//unsigned long *blue_average;

	/* Allocate Device Memory */
	cudaMalloc((void **)&d_c, sizeof(unsigned int));
	// Red
	cudaMalloc((void **)&d_red, sizeof(unsigned short**) * height);
	cudaMalloc((void **)&d_red_data, sizeof(unsigned short) * height * width);
	// Green
	cudaMalloc((void **)&d_green, sizeof(unsigned short**) * height);
	cudaMalloc((void **)&d_green_data, sizeof(unsigned short) * height * width);
	// Blue
	cudaMalloc((void **)&d_blue, sizeof(unsigned short**) * height);
	cudaMalloc((void **)&d_blue_data, sizeof(unsigned short) * height * width);
	// Average
	//cudaMalloc((void **)&red_average, sizeof(unsigned long));
	//cudaMalloc((void **)&green_average, sizeof(unsigned long));
	//cudaMalloc((void **)&blue_average, sizeof(unsigned long));
	checkCUDAError("Memory allocation");

	/* Allocate Host Memory */
	for (int i = 0; i < height; i++) {
		// Input
		h_red[i] = d_red_data + width * i;
		h_green[i] = d_green_data + width * i;
		h_blue[i] = d_blue_data + width * i;
	}

	/* Copy Host Input to Device Input */
	// Red
	cudaMemcpy(d_red, h_red, sizeof(unsigned short *) * height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_red_data, red_vector, sizeof(unsigned short) * height * width, cudaMemcpyHostToDevice);
	// Green
	cudaMemcpy(d_green, h_green, sizeof(unsigned short *) * height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_green_data, green_vector, sizeof(unsigned short) * height * width, cudaMemcpyHostToDevice);
	// Blue
	cudaMemcpy(d_blue, h_blue, sizeof(unsigned short *) * height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_blue_data, blue_vector, sizeof(unsigned short) * height * width, cudaMemcpyHostToDevice);
	// C
	cudaMemcpy(d_c, &c, sizeof(unsigned int), cudaMemcpyHostToDevice);
	// Average
	cudaMemcpyToSymbol(red_average, &h_red_average, sizeof(unsigned long), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(green_average, &h_green_average, sizeof(unsigned long), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(blue_average, &h_blue_average, sizeof(unsigned long), cudaMemcpyHostToDevice);
	checkCUDAError("Input transfer to device");

	/* Configure the Grid of Thread Blocks and Run the GPU Kernel */
	// Single Block
	dim3 blocksPerGrid(1, 1, 1);
	dim3 threadsPerBlock(c, c, 1);
	// Start Timing
	cudaEventRecord(start);
	assignCell << < blocksPerGrid, threadsPerBlock >> > (d_c, d_red, d_green, d_blue);
	cudaEventRecord(stop);

	/* Wait for All Threads to Complete */
	cudaThreadSynchronize();
	// Stop Timing
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	checkCUDAError("Kernel execution");

	/* Copy the GPU Output back to the Host */
	cudaMemcpy(red_vector, d_red_data, sizeof(unsigned short) * height * width, cudaMemcpyDeviceToHost);
	cudaMemcpy(green_vector, d_green_data, sizeof(unsigned short) * height * width, cudaMemcpyDeviceToHost);
	cudaMemcpy(blue_vector, d_blue_data, sizeof(unsigned short) * height * width, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&h_red_average, red_average, sizeof(unsigned long), cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&h_green_average, green_average, sizeof(unsigned long), cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&h_blue_average, blue_average, sizeof(unsigned long), cudaMemcpyDeviceToHost);
	checkCUDAError("Result transfer to host");
	vec2matrix();

	/* Free Device Memory */
	// Red
	cudaFree(d_red);
	cudaFree(d_red_data);
	// Green
	cudaFree(d_green);
	cudaFree(d_green_data);
	// Bluie
	cudaFree(d_blue);
	cudaFree(d_blue_data);
	// Average
	//cudaFree(red_average);
	//cudaFree(green_average);
	//cudaFree(blue_average);
	checkCUDAError("Free memory");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("|| CUDA Average Image Colour\n");
	printf("|| -- red = %hu\n", h_red_average / (width*height));
	printf("|| -- green = %hu\n", h_green_average / (width*height));
	printf("|| -- blue = %hu\n", h_blue_average / (width*height));
	printf("|| CUDA Mode Execution Time\n");
	printf("|| -- %.0f s\n", milliseconds / 1000.0);
	printf("|| -- %.10f ms\n", milliseconds);
	printf("=============== Stop Run CUDA! ===============\n");
}

/*
  CPU mode
*/
int runCPU()
{
	printf("\n=============== Start Run CPU! ===============\n");
	/* Set Clock */
	double begin_cpu, end_cpu;
	// Start Timing
	begin_cpu = omp_get_wtime();

	/* Initialise Average */
	unsigned short red_average = 0;
	unsigned short blue_average = 0;
	unsigned short green_average = 0;

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
			printf("%d-%d: %d-%d-%d\n", i, j, red_average_all, green_average_all, blue_average_all);

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
	red_average = (unsigned short)red_average_all;
	green_average = (unsigned short)green_average_all;
	blue_average = (unsigned short)blue_average_all;

	// Stop Timing
	end_cpu = omp_get_wtime();
	printf("|| CPU Average Image Colour\n");
	printf("|| -- red = %hu\n", red_average);
	printf("|| -- green = %hu\n", green_average);
	printf("|| -- blue = %hu\n", blue_average);
	printf("|| CPU Mode Execution Time\n");
	printf("|| -- %.0f s\n", (end_cpu - begin_cpu));
	printf("|| -- %.5f ms\n", (end_cpu - begin_cpu)*1000.0);
	printf("=============== Stop Run CPU! ===============\n");
	return 1;
}

/*
  OPENMP
*/
int runOPENMP() {
	printf("\n=============== Start Run OPENMP! ===============\n");
	/* Set Clock */
	double begin_openmp, end_openmp;
	// Start Timing
	begin_openmp = omp_get_wtime();

	/* Initialise Average */
	unsigned short red_average = 0;
	unsigned short blue_average = 0;
	unsigned short green_average = 0;

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
	red_average = (unsigned short)red_average_all;
	green_average = (unsigned short)green_average_all;
	blue_average = (unsigned short)blue_average_all;

	// Stop Timing
	end_openmp = omp_get_wtime();
	printf("|| OPENMP Average Image Colour\n");
	printf("|| -- red = %hu\n", red_average);
	printf("|| -- green = %hu\n", green_average);
	printf("|| -- blue = %hu\n", blue_average);
	printf("|| OPENMP Mode Execution Time\n");
	printf("|| -- %.0f s\n", (end_openmp - begin_openmp));
	printf("|| -- %.5f ms\n", (end_openmp - begin_openmp)*1000.0);
	printf("=============== Stop Run OPENMP! ===============\n");
	return 1;
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
	printf("\n=============== Read in Arguments ===============\n");

	// first argument is always the executable name
	printf("|| -- Executable Name: %s\n", argv[0]);

	// read in the non optional command line arguments
	printf("|| -- c: %s\n", argv[1]);
	for (int i = 0; i < strlen(argv[1]); i++) {
		if ((argv[1][i] >= 'a' && argv[1][i] <= 'z') || (argv[1][i] >= 'A' && argv[1][i] <= 'Z') || argv[1][i] == '-' || argv[1][i] == '.')
		{
			printf("=============== Read in Stop! ===============\n");
			fprintf(stderr, "Error: Wrong c argument. C should be any positive integer.\n");
			printf("=============================================\n");
			return FAILURE;
		}
	}
	if (atoi(argv[1]) == 0) {
		printf("=============== Read in Stop! ===============\n");
		fprintf(stderr, "Error: Wrong c argument. C should not be zero.\n");
		printf("=============================================\n");
		return FAILURE;
	}
	else {
		if (atoi(argv[1]) % 2 == 0) {
			c = (unsigned int)atoi(argv[1]);
		}
		else {
			printf("=============== Read in Stop! ===============\n");
			fprintf(stderr, "Error: Wrong c argument. C should a power of 2 number.\n");
			printf("=============================================\n");
			return FAILURE;
		}
	}
	

	// read in the mode
	printf("|| -- Mode: %s\n", argv[2]);
	if (!strcmp(argv[2], "CPU"))
		execution_mode = CPU;
	else if (!strcmp(argv[2], "OPENMP"))
		execution_mode = OPENMP;
	else if (!strcmp(argv[2], "CUDA"))
		execution_mode = CUDA;
	else if (!strcmp(argv[2], "ALL"))
		execution_mode = ALL;
	else {
		printf("=============== Read in Stop! ===============\n");
		fprintf(stderr, "Error: Wrong mode argument. Correct usage is CPU, OPENMP, CUDA or ALL.\n");
		printf("=============================================\n");
		return FAILURE;
	}

	// read in the input image name
	printf("|| -- Input Image Name: %s\n", argv[4]);
	input_image_name = argv[4];

	// read in the output image name
	printf("|| -- Output Image Name: %s\n", argv[6]);
	output_image_name = argv[6];

	// read in any optional part 3 arguments
	if (argc == 9) {
		printf("\n+++++++++++++++ Read in Optional Part +++++++++++++++\n");
		printf("|| -- output Image Format: %s\n", argv[8]);
		if (!strcmp(argv[8], "PPM_BINARY"))
			image_format = PPM_BINARY;
		else if (!strcmp(argv[8], "PPM_PLAIN_TEXT"))
			image_format = PPM_PLAIN_TEXT;
		else {
			printf("=============== Read in Stop! ===============\n");
			fprintf(stderr, "Error: Wrong image format argument. Correct usage is PPM_BINARY or PPM_PLAIN_TEXT.\n");
			printf("=============================================\n");
			return FAILURE;
		}
	}
	printf("=============== Read in Complete! ===============\n");
	return SUCCESS;
}

int readFile() {
	/* cache */
	char temp[3]; // temporarily store each read result 
	char *temp_value = (char *)malloc(sizeof(char) * 4); // temporarily store the reading results

	FILE *f = NULL;

	// open file as binary
	printf("\n=============== Start Read Input File ===============\n");
	f = fopen(input_image_name, "rb");

	if (f == NULL) {
		printf("=============== Stop Read Input File! ===============\n");
		fprintf(stderr, "Could not open input file: %s\n", input_image_name);
		printf("=====================================================\n");
		exit(1);
	}


	// Read magic number
	fread(&magic_number, sizeof(char), 3, f);
	printf("|| -- Magic Number: %s\n", magic_number);
	magic_number[2] = '\0';
	if (strncmp(magic_number, "P3", 2) == 0) {
		// P3 - Plain Text
		printf("---------- Start Read Plain Text File ----------\n");

		for (int i = 0; i < 3; i++) {
			fgets(comment, 64, f);

			// skip the comment part
			if (strncmp(comment, "#", 1) == 0) {
				printf("|| -- Comment is: %s\n", comment);
				i--;
			}
			else if (width == 0) {
				width = (unsigned short)atoi(comment);
				printf("|| -- Width is: %d\n", width);
			}	
			else if (height == 0) {
				height = (unsigned short)atoi(comment);
				printf("|| -- Height is: %d\n", height);
			}
			else {
				max_color_value = (unsigned short)atoi(comment);
				printf("|| -- Max Color Value is: %d\n", max_color_value);
			}
		}

		unsigned short tempWidth = 0;
		unsigned short tempHeight = 0;

		// Check Error
		if ((width == 0) || (height == 0)) {
			printf("=============== Stop Read Input File! ===============\n");
			fprintf(stderr, "Width or Height is Zero. Please Check it.\n");
			printf("=====================================================\n");
			exit(1);
		}
		else {
			tempWidth = width;
			tempHeight = height;
		}

		// allocate memory for arrays
		allocateMemory();

		// assign value for arrays
		for (int i = 0; i < width * height; i++) {
			if (fscanf(f, "%hu %hu %hu", &red_vector[i], &green_vector[i], &blue_vector[i])) {
				tempWidth--;
				tempHeight--;
				if ((red_vector[i] < 0) && (red_vector[i] > max_color_value)) {
					printf("=============== Stop Read Input File! ===============\n");
					fprintf(stderr, "Input file is broken. The value of red pixel is wrong.\n");
					printf("=====================================================\n");
					exit(1);
				}
				if ((green_vector[i] < 0) && (green_vector[i] > max_color_value)) {
					printf("=============== Stop Read Input File! ===============\n");
					fprintf(stderr, "Input file is broken. The value of green pixel is wrong.\n");
					printf("=====================================================\n");
					exit(1);
				}
				if ((blue_vector[i] < 0) && (blue_vector[i] > max_color_value)) {
					printf("=============== Stop Read Input File! ===============\n");
					fprintf(stderr, "Input file is broken. The value of blue pixel is wrong.\n");
					printf("=====================================================\n");
					exit(1);
				}
			}
			else {
				printf("=============== Stop Read Input File! ===============\n");
				fprintf(stderr, "Input file is broken. The numbe of pixel is wrong.\n");
				printf("=====================================================\n");
				exit(1);
			}
		}
		vec2matrix();
		printf("=============== Read Plain Text File is Finished! ===============\n");
	}
	else if (strncmp(magic_number, "P6", 2) == 0) {
		// P6 - Binary File
		printf("---------- Start Read Binary File ----------\n");
		int i = 0;
		int j = 0;


		// read in width
		while (1) {
			fread(temp, 1, 1, f);
			if (!strncmp("\n", temp, 1))
				break;
			else
				temp_value[i++] = temp[0];
		}
		temp_value[i] = '\0';
		width = atoi(temp_value);
		printf("|| -- Width is: %d\n", width);


		// read in height
		i = 0;
		while (1) {
			fread(temp, 1, 1, f);
			if (!strncmp("\n", temp, 1))
				break;
			else
				temp_value[i++] = temp[0];
		}
		temp_value[i] = '\0';
		height = atoi(temp_value);
		printf("|| -- Height is: %d\n", height);


		// read in max color value
		i = 0;
		while (1) {
			fread(temp, 1, 1, f);
			if (!strncmp("\n", temp, 1))
				break;
			else
				temp_value[i++] = temp[0];
		}
		temp_value[i] = '\0';
		max_color_value = atoi(temp_value);
		printf("|| -- Max Color Value is: %d\n", max_color_value);

		// allocate memory for arrays
		allocateMemory();

		unsigned char *color_temp = (unsigned char*)malloc(sizeof(char) * 3 * width * height);
		fread(color_temp, 3, width * height, f);

		// read in red, green and blue into 3 arrays.
		for (i = 0; i < height * width; i++) {
			red_vector[i] = *(color_temp + i * 3);
			green_vector[i] = *(color_temp + i * 3 + 1);
			blue_vector[i] = *(color_temp + i * 3 + 2);
		}
		vec2matrix();
		printf("=============== Read Binary File is Finished! ===============\n");
		free(color_temp);
	}
	else {
	printf("=============== Stop Read Input File! ===============\n");
		fprintf(stderr, "Magic Number is not P3 or P6: %s\n", input_image_name);
		printf("=====================================================\n");
		exit(1);
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

	red_vector = (unsigned short *)malloc(sizeof(unsigned short) * width * height);
	green_vector = (unsigned short *)malloc(sizeof(unsigned short) * width * height);
	blue_vector = (unsigned short *)malloc(sizeof(unsigned short) * width * height);

	return 1;
}

int allocateCUDAMemory(unsigned short **d_red_input, unsigned short **d_red_output, unsigned short **d_green_input, unsigned short **d_green_output, unsigned short **d_blue_input, unsigned short **d_blue_output) {
	/* Red */
	cudaMalloc((void ***)&d_red_input, sizeof(unsigned short *) * height);
	for (int i = 0; i < height; i++)
		cudaMalloc((void **)&d_red_input[i], sizeof(unsigned short) * width);
	cudaMalloc((void ***)&d_red_output, sizeof(unsigned short *) * height);
	for (int i = 0; i < height; i++)
		cudaMalloc((void **)&d_red_output[i], sizeof(unsigned short) * width);

	/* Green */
	cudaMalloc((void ***)&d_green_input, sizeof(unsigned short *) * height);
	for (int i = 0; i < height; i++)
		cudaMalloc((void **)&d_green_input[i], sizeof(unsigned short) * width);
	cudaMalloc((void ***)&d_green_output, sizeof(unsigned short *) * height);
	for (int i = 0; i < height; i++)
		cudaMalloc((void **)&d_green_output[i], sizeof(unsigned short) * width);

	/* Blue */
	cudaMalloc((void ***)&d_blue_input, sizeof(unsigned short *) * height);
	for (int i = 0; i < height; i++)
		cudaMalloc((void **)&d_blue_input[i], sizeof(unsigned short) * width);
	cudaMalloc((void ***)&d_blue_output, sizeof(unsigned short *) * height);
	for (int i = 0; i < height; i++)
		cudaMalloc((void **)&d_blue_output[i], sizeof(unsigned short) * width);

	return 1;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

/*
  Convert vector to matrix
*/
void vec2matrix() 
{
	int k = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			red[i][j] = red_vector[k];
			green[i][j] = green_vector[k];
			blue[i][j] = blue_vector[k];
			k++;
		}
	}
}

/*
  write to binary file
*/
int writeBinary() {
	int i, j;
	FILE *f = NULL;

	printf("========== Start write binary file ==========\n");

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
	printf("========== Write binary file is finished! ==========\n");

	return 1;
}

/*
  write to plain text file
*/
int writePlainText() {
	int i, j;
	FILE *f = NULL;

	printf("\n========== Start write plain text file ==========\n");

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
	printf("========== Write plain text file is finished!==========\n");

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