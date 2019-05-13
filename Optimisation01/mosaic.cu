#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <vector_types.h>
#include <vector_functions.h>

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
void matrix2cellvec();
void cellvec2matrix();

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

/* pixel value - Cell Vector */
unsigned short *red_cell_vector; // put cell value together
unsigned short *green_cell_vector;
unsigned short *blue_cell_vector;

/* pixel value - Cell Vector - Index*/
unsigned long *cell_vector_index; // start and end point

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

	vec2matrix();
	matrix2cellvec();

	// Execute the mosaic filter based on the mode
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
		cellvec2matrix();
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
		/*switch (image_format) {
		case (PPM_BINARY): {
			writeBinary();
			break;
		}
		case (PPM_PLAIN_TEXT): {
			writePlainText();
			break;
		}
		}*/


		/* OPENMP */
		readFile();
		vec2matrix();
		// calculate the average colour value
		runOPENMP();

		// Save the output image file (from last executed mode)
		/*switch (image_format) {
		case (PPM_BINARY): {
			writeBinary();
			break;
		}
		case (PPM_PLAIN_TEXT): {
			writePlainText();
			break;
		}
		}*/

		/* CUDA */
		readFile();
		vec2matrix();
		runCUDA();

		cellvec2matrix();

		// Save the output image file (from last executed mode)
		/*switch (image_format) {
		case (PPM_BINARY): {
			writeBinary();
			break;
		}
		case (PPM_PLAIN_TEXT): {
			writePlainText();
			break;
		}
		}*/

		break;
	}
	}

	freeMemory();
	//getchar();
	cudaDeviceReset();
	return 0;
}

/*
  CUDA
*/
__constant__ unsigned short D_C;
__constant__ unsigned short QUOTIENT_ROW;
__constant__ unsigned short REMAINDER_ROW;
__constant__ unsigned short QUOTIENT_COLUMN;
__constant__ unsigned short REMAINDER_COLUMN;
__constant__ unsigned short CELLS_PER_COLUMN;

__device__ int d_red_sum;
__device__ int d_green_sum;
__device__ int d_blue_sum;

__device__ unsigned short d_red;
__device__ unsigned short d_green;
__device__ unsigned short d_blue;

texture<unsigned long, cudaTextureType1D, cudaReadModeElementType> d_cell_index;

__device__ void computeCell(unsigned long pixel_num, long start_point, long end_point, unsigned short *d_color, int *d_color_sum_local) {
	unsigned long average_pixel = 0;
	int i, j;
	for (i = start_point; i < end_point; i++) {
		average_pixel += d_color[i];
	}
	atomicAdd(&d_color_sum_local[0], average_pixel);
	average_pixel /= pixel_num;
	for (i = start_point; i < end_point; i++) {
		d_color[i] = average_pixel;
	}
}

__global__ void assignCell(unsigned short *d_red_cell_vector, unsigned short *d_green_cell_vector, unsigned short *d_blue_cell_vector)
{
	// blockIdx.x --- the row number of the cell
	// blockIdx.y --- the column number of the cell
	long start_point = tex1Dfetch(d_cell_index, (blockIdx.x * CELLS_PER_COLUMN + blockIdx.y) * 2);
	long end_point = tex1Dfetch(d_cell_index, 1 + (blockIdx.x * CELLS_PER_COLUMN + blockIdx.y) * 2);
	long pixel_num = end_point - start_point;
	// convert global data pointer to the local pointer of this block

	/* Retrieve Global Variables */
	int *d_red_sum_local = &d_red_sum;
	int *d_green_sum_local = &d_green_sum;
	int *d_blue_sum_local = &d_blue_sum;

	/* the width and height for a cell */
	// change limitaion height of the cell 
	unsigned short limitation_height = (blockIdx.x == QUOTIENT_COLUMN) ? REMAINDER_COLUMN : D_C;

	// change limitation width of the cell
	unsigned short limitation_width = (blockIdx.y == QUOTIENT_ROW) ? REMAINDER_ROW : D_C;

	// Red
	if (blockIdx.z == 0) {
		computeCell(pixel_num, start_point, end_point, d_red_cell_vector, d_red_sum_local);
	}

	// Green
	if (blockIdx.z == 1) {
		computeCell(pixel_num, start_point, end_point, d_green_cell_vector, d_green_sum_local);
	}

	// Blue
	if (blockIdx.z == 2) {
		computeCell(pixel_num, start_point, end_point, d_blue_cell_vector, d_blue_sum_local);
	}
	//printf("%d-%d: %d-%d-%d\n", blockIdx.x, threadIdx.x, d_sum_red_row[blockIdx.x], d_sum_green_row[blockIdx.x], d_sum_blue_row[blockIdx.x]);
	//printf("-- Global --- %d-%d: %d-%d-%d\n", threadIdx.x, threadIdx.y, d_sum_red_row[blockIdx.x], d_sum_green_row[blockIdx.x], d_sum_blue_row[blockIdx.x]);
}


void runCUDA()
{
	printf("\n=============== Start Run CUDA! ===============\n");
	/* Set Clock */
	cudaEvent_t start, start_core, stop, stop_core;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_core);
	cudaEventCreate(&stop_core);
	// Start Timing Total
	cudaEventRecord(start);

	/* Set  */
	unsigned int pixels_per_row = width;
	unsigned int pixels_per_column = height;
	/* the number and width of cells */
	unsigned short quotient_row = width / c; // the number of square cells in a row
	unsigned short remainder_row = width % c; // the width of the rest cell in a row (optional)
	unsigned short quotient_column = height / c; // the number of square cells in a column
	unsigned short remainder_column = height % c; // the height of the rest cell in a column (optional)

	/* the total number of cells in a row and column */
	unsigned short cells_per_row = (remainder_row) ? (quotient_row + 1) : quotient_row; // the number of cells in a row (decide i)
	unsigned short cells_per_column = (remainder_column) ? (quotient_column + 1) : quotient_column; // the number of cells in a column (decide j)

	/* the width and height for a cell */
	unsigned short limitation_width; // the width of a cell
	unsigned short limitation_height; // the height of a cell

	unsigned long *d_cell_vector_index;
	// Red
	unsigned short *d_red_cell_vector;
	//Green
	unsigned short *d_green_cell_vector;
	// Blue
	unsigned short *d_blue_cell_vector;

	int *h_red_sum = (int *)malloc(sizeof(int));
	*h_red_sum = 0;
	int *h_green_sum = (int *)malloc(sizeof(int));
	*h_green_sum = 0;
	int *h_blue_sum = (int *)malloc(sizeof(int));
	*h_blue_sum = 0;

	/* Allocate Device Memory */
	unsigned long d_cell_vector_size = sizeof(unsigned short) * width * height;
	unsigned long d_cell_vector_index_size = sizeof(unsigned long) * cells_per_row * cells_per_column * 2;
	cudaMalloc((void **)&d_cell_vector_index, d_cell_vector_index_size);
	// Red
	cudaMalloc((void **)&d_red_cell_vector, d_cell_vector_size);
	// Green
	cudaMalloc((void **)&d_green_cell_vector, d_cell_vector_size);
	// Blue
	cudaMalloc((void **)&d_blue_cell_vector, d_cell_vector_size);
	checkCUDAError("Memory allocation");

	/* Copy Host Input to Device Input */
	cudaMemcpy(d_cell_vector_index, cell_vector_index, d_cell_vector_index_size, cudaMemcpyHostToDevice);
	// Red
	cudaMemcpy(d_red_cell_vector, red_cell_vector, d_cell_vector_size, cudaMemcpyHostToDevice);
	// Green
	cudaMemcpy(d_green_cell_vector, green_cell_vector, d_cell_vector_size, cudaMemcpyHostToDevice);
	// Blue
	cudaMemcpy(d_blue_cell_vector, blue_cell_vector, d_cell_vector_size, cudaMemcpyHostToDevice);

	/* Copy to Constant */
	cudaMemcpyToSymbol(D_C, &c, sizeof(unsigned short));
	/* the number and width of cells */
	cudaMemcpyToSymbol(QUOTIENT_ROW, &quotient_row, sizeof(unsigned short)); // the number of square cells in a row
	cudaMemcpyToSymbol(REMAINDER_ROW, &remainder_row, sizeof(unsigned short)); // the width of the rest cell in a row (optional)
	cudaMemcpyToSymbol(QUOTIENT_COLUMN, &quotient_column, sizeof(unsigned short)); // the number of square cells in a column
	cudaMemcpyToSymbol(REMAINDER_COLUMN, &remainder_column, sizeof(unsigned short)); // the height of the rest cell in a column (optional)
	cudaMemcpyToSymbol(CELLS_PER_COLUMN, &cells_per_column, sizeof(unsigned short));
	checkCUDAError("Input transfer to device");

	cudaBindTexture(0, d_cell_index, d_cell_vector_index, d_cell_vector_index_size);
	checkCUDAError("tex1D bind");

	/* Configure the Grid of Thread Blocks and Run the GPU Kernel */
	// Single Block
	dim3 blocksPerGrid(cells_per_column, cells_per_row, 3);
	dim3 threadsPerBlock(1, 1, 1);
	//printf("%d-%d\n", cell_per_column, cell_per_row);

	// Start Timing Core
	cudaEventRecord(start_core);
	assignCell << < blocksPerGrid, threadsPerBlock >> > (d_red_cell_vector, d_green_cell_vector, d_blue_cell_vector);

	/* Wait for All Threads to Complete */
	cudaThreadSynchronize();
	checkCUDAError("Kernel execution");
	// Stop Timing Core
	cudaEventRecord(stop_core);
	float milliseconds_core = 0;
	cudaEventSynchronize(stop_core);
	cudaEventElapsedTime(&milliseconds_core, start_core, stop_core);

	/* Copy the GPU Output back to the Host */
	cudaMemcpy(red_cell_vector, d_red_cell_vector, d_cell_vector_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(green_cell_vector, d_green_cell_vector, d_cell_vector_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(blue_cell_vector, d_blue_cell_vector, d_cell_vector_size, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(h_red_sum, d_red_sum, sizeof(int));
	cudaMemcpyFromSymbol(h_green_sum, d_green_sum, sizeof(int));
	cudaMemcpyFromSymbol(h_blue_sum, d_blue_sum, sizeof(int));
	checkCUDAError("Result transfer to host");

	int red_average = *h_red_sum / (pixels_per_column * pixels_per_row);
	int green_average = *h_green_sum / (pixels_per_column * pixels_per_row);
	int blue_average = *h_blue_sum / (pixels_per_column * pixels_per_row);

	/* Free Device Memory */
	cudaFree(d_cell_vector_index);
	// Red
	cudaFree(d_red_cell_vector);
	// Green
	cudaFree(d_green_cell_vector);
	// Bluie
	cudaFree(d_blue_cell_vector);
	checkCUDAError("Free memory");

	/* Free Host Memory */
	free(h_red_sum);
	free(h_green_sum);
	free(h_blue_sum);

	// Stop Timing Total
	cudaEventRecord(stop);
	float milliseconds = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(start_core);
	cudaEventDestroy(stop_core);


	printf("|| CUDA Average Image Colour\n");
	printf("|| -- red = %hu\n", red_average);
	printf("|| -- green = %hu\n", green_average);
	printf("|| -- blue = %hu\n", blue_average);
	printf("|| CUDA Mode Total Execution Time\n");
	printf("|| -- %.0f s\n", milliseconds / 1000.0);
	printf("|| -- %.10f ms\n", milliseconds);
	printf("|| CUDA Mode Core Part Execution Time\n");
	printf("|| -- %.0f s\n", milliseconds_core / 1000.0);
	printf("|| -- %.10f ms\n", milliseconds_core);
	printf("=============== Stop Run CUDA! ===============\n");
}

/*
  CPU mode
*/
int runCPU()
{
	printf("\n=============== Start Run CPU! ===============\n");
	/* Set Clock */
	double begin_cpu, begin_cpu_core, end_cpu, end_cpu_core;
	// Start Timing Total
	begin_cpu = omp_get_wtime();

	/* Initialise Average */
	unsigned short red_average = 0;
	unsigned short blue_average = 0;
	unsigned short green_average = 0;

	/* iteration */
	unsigned short i, j, k, l = 0;

	/* Set  */
	/* the number and width of cells */
	unsigned short quotient_row = width / c; // the number of square cells in a row
	unsigned short remainder_row = width % c; // the width of the rest cell in a row (optional)
	unsigned short quotient_column = height / c; // the number of square cells in a column
	unsigned short remainder_column = height % c; // the height of the rest cell in a column (optional)

	/* the total number of cells in a row and column */
	unsigned short cells_per_row = (remainder_row) ? (quotient_row + 1) : quotient_row; // the number of cells in a row (decide i)
	unsigned short cells_per_column = (remainder_column) ? (quotient_column + 1) : quotient_column; // the number of cells in a column (decide j)

	/* the width and height for a cell */
	unsigned short limitation_width; // the width of a cell
	unsigned short limitation_height; // the height of a cell

	/* the total number of cells */
	int cells = cells_per_row * cells_per_column;

	/* average value in a cell */
	unsigned long red_average_part = 0;
	unsigned long green_average_part = 0;
	unsigned long blue_average_part = 0;

	/* average value in all */
	unsigned long red_average_all = 0;
	unsigned long green_average_all = 0;
	unsigned long blue_average_all = 0;

	// Start Timing COre
	begin_cpu_core = omp_get_wtime();
	for (i = 0; i < cells_per_column; i++) {
		// loop cells in column

		// change limitaion height of the cell 
		limitation_height = (i == quotient_column) ? remainder_column : c;

		for (j = 0; j < cells_per_row; j++) {
			// loop cells in row

			// initial average in a cell
			red_average_part = 0;
			green_average_part = 0;
			blue_average_part = 0;

			// change limitation width of the cell
			limitation_width = (j == quotient_row) ? remainder_row : c;

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
			//printf("%d-%d: %d-%d-%d\n", i, j, red_average_all, green_average_all, blue_average_all);

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
	// Stop Timing Core
	end_cpu_core = omp_get_wtime();

	// calculate all average pixel values
	red_average_all /= (width * height);
	green_average_all /= (width * height);
	blue_average_all /= (width * height);

	// return pixel values
	red_average = (unsigned short)red_average_all;
	green_average = (unsigned short)green_average_all;
	blue_average = (unsigned short)blue_average_all;

	// Stop Timing Total
	end_cpu = omp_get_wtime();
	printf("|| CPU Average Image Colour\n");
	printf("|| -- red = %hu\n", red_average);
	printf("|| -- green = %hu\n", green_average);
	printf("|| -- blue = %hu\n", blue_average);
	printf("|| CPU Mode Total Execution Time\n");
	printf("|| -- %.0f s\n", (end_cpu - begin_cpu));
	printf("|| -- %.5f ms\n", (end_cpu - begin_cpu)*1000.0);
	printf("|| CPU Mode Core Part Execution Time\n");
	printf("|| -- %.0f s\n", (end_cpu_core - begin_cpu_core));
	printf("|| -- %.5f ms\n", (end_cpu_core - begin_cpu_core)*1000.0);
	printf("=============== Stop Run CPU! ===============\n");
	return 1;
}

/*
  OPENMP
*/
int runOPENMP() {
	printf("\n=============== Start Run OPENMP! ===============\n");
	/* Set Clock */
	double begin_openmp, begin_openmp_core, end_openmp, end_openmp_core;
	// Start Timing Total
	begin_openmp = omp_get_wtime();

	/* Initialise Average */
	unsigned short red_average = 0;
	unsigned short blue_average = 0;
	unsigned short green_average = 0;

	/* iteration */
	signed short i, j, k, l = 0;
	/* Set  */
	/* the number and width of cells */
	unsigned short quotient_row = width / c; // the number of square cells in a row
	unsigned short remainder_row = width % c; // the width of the rest cell in a row (optional)
	unsigned short quotient_column = height / c; // the number of square cells in a column
	unsigned short remainder_column = height % c; // the height of the rest cell in a column (optional)

	/* the total number of cells in a row and column */
	unsigned short cells_per_row = (remainder_row) ? (quotient_row + 1) : quotient_row; // the number of cells in a row (decide i)
	unsigned short cells_per_column = (remainder_column) ? (quotient_column + 1) : quotient_column; // the number of cells in a column (decide j)

	/* the width and height for a cell */
	unsigned short limitation_width; // the width of a cell
	unsigned short limitation_height; // the height of a cell

	/* the total number of cells */
	int cells = cells_per_row * cells_per_column;

	/* average value in a cell */
	unsigned long red_average_part = 0;
	unsigned long green_average_part = 0;
	unsigned long blue_average_part = 0;

	/* average value in all */
	unsigned long red_average_all = 0;
	unsigned long green_average_all = 0;
	unsigned long blue_average_all = 0;

	// Start Timing Total
	begin_openmp_core = omp_get_wtime();
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
	// Stop Timing Core
	end_openmp_core = omp_get_wtime();

	// calculate all average pixel values
	red_average_all /= width * height;
	green_average_all /= width * height;
	blue_average_all /= width * height;

	// return pixel values
	red_average = (unsigned short)red_average_all;
	green_average = (unsigned short)green_average_all;
	blue_average = (unsigned short)blue_average_all;

	// Stop Timing Total
	end_openmp = omp_get_wtime();
	printf("|| OPENMP Average Image Colour\n");
	printf("|| -- red = %hu\n", red_average);
	printf("|| -- green = %hu\n", green_average);
	printf("|| -- blue = %hu\n", blue_average);
	printf("|| OPENMP Mode Total Execution Time\n");
	printf("|| -- %.0f s\n", (end_openmp - begin_openmp));
	printf("|| -- %.5f ms\n", (end_openmp - begin_openmp)*1000.0);
	printf("|| OPENMP Mode Core Part Execution Time\n");
	printf("|| -- %.0f s\n", (end_openmp_core - begin_openmp_core));
	printf("|| -- %.5f ms\n", (end_openmp_core - begin_openmp_core)*1000.0);
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
	char *temp_value = (char *)malloc(sizeof(char) * 10); // temporarily store the reading results

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
		printf("=============== Read Binary File is Finished! ===============\n");
		free(color_temp);
	}
	else {
		printf("=============== Stop Read Input File! ===============\n");
		fprintf(stderr, "Magic Number is not P3 or P6: %s\n", input_image_name);
		printf("=====================================================\n");
		exit(1);
	}

	free(temp_value);
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
  Convert Matrix to Vector in Cell's Order

*/
void matrix2cellvec()
{
	int h = 0;
	/* Set  */
	/* the number and width of cells */
	unsigned short quotient_row = width / c; // the number of square cells in a row
	unsigned short remainder_row = width % c; // the width of the rest cell in a row (optional)
	unsigned short quotient_column = height / c; // the number of square cells in a column
	unsigned short remainder_column = height % c; // the height of the rest cell in a column (optional)

	/* the total number of cells in a row and column */
	unsigned short cells_per_row = (remainder_row) ? (quotient_row + 1) : quotient_row; // the number of cells in a row (decide i)
	unsigned short cells_per_column = (remainder_column) ? (quotient_column + 1) : quotient_column; // the number of cells in a column (decide j)

	/* the width and height for a cell */
	unsigned short cell_width; // the width of a cell
	unsigned short cell_height; // the height of a cell

	/* Allocate Memory */
	red_cell_vector = (unsigned short *)malloc(sizeof(unsigned short) * width * height);
	green_cell_vector = (unsigned short *)malloc(sizeof(unsigned short) * width * height);
	blue_cell_vector = (unsigned short *)malloc(sizeof(unsigned short) * width * height);

	cell_vector_index = (unsigned long *)malloc(sizeof(unsigned long) * cells_per_row * cells_per_column * 2);

	for (int i = 0; i < cells_per_column; i++) {
		// change limitaion height of the cell 
		cell_height = (i == quotient_column) ? remainder_column : c;
		for (int j = 0; j < cells_per_row; j++) {
			// change limitation width of the cell
			cell_width = (j == quotient_row) ? remainder_row : c;

			// Start Point
			cell_vector_index[(i*cells_per_column + j) * 2] = h;

			// sum all pixel values in a cell
			for (int k = 0; k < cell_height; k++) {
				// loop pixel in cell row
				for (int l = 0; l < cell_width; l++) {
					// loop pixel in cell column

					// sum up pixel values
					red_cell_vector[h] = red[i*c + k][j*c + l];
					green_cell_vector[h] = green[i*c + k][j*c + l];
					blue_cell_vector[h] = blue[i*c + k][j*c + l];
					h++;
				}
			}

			// End Point
			cell_vector_index[(i*cells_per_column + j) * 2 + 1] = h;
		}
	}
}

void cellvec2matrix() {
	int h = 0;
	/* Set  */
	/* the number and width of cells */
	unsigned short quotient_row = width / c; // the number of square cells in a row
	unsigned short remainder_row = width % c; // the width of the rest cell in a row (optional)
	unsigned short quotient_column = height / c; // the number of square cells in a column
	unsigned short remainder_column = height % c; // the height of the rest cell in a column (optional)

	/* the total number of cells in a row and column */
	unsigned short cells_per_row = (remainder_row) ? (quotient_row + 1) : quotient_row; // the number of cells in a row (decide i)
	unsigned short cells_per_column = (remainder_column) ? (quotient_column + 1) : quotient_column; // the number of cells in a column (decide j)

	/* the width and height for a cell */
	unsigned short cell_width; // the width of a cell
	unsigned short cell_height; // the height of a cell

	for (int i = 0; i < cells_per_column; i++) {
		// change limitaion height of the cell 
		cell_height = (i == quotient_column) ? remainder_column : c;
		for (int j = 0; j < cells_per_row; j++) {
			// change limitation width of the cell
			cell_width = (j == quotient_row) ? remainder_row : c;

			// sum all pixel values in a cell
			for (int k = 0; k < cell_height; k++) {
				// loop pixel in cell row
				for (int l = 0; l < cell_width; l++) {
					// loop pixel in cell column

					// sum up pixel values
					red[i*c + k][j*c + l] = red_cell_vector[h];
					green[i*c + k][j*c + l] = green_cell_vector[h];
					blue[i*c + k][j*c + l] = blue_cell_vector[h];
					h++;
				}
			}
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

	free(red_vector);
	free(green_vector);
	free(blue_vector);

	free(red_cell_vector);
	free(green_cell_vector);
	free(blue_cell_vector);

	free(cell_vector_index);

	return 1;
}
