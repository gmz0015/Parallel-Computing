#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//The number of character in the encrypted text
#define N 1024
#define A 15
#define B 27
#define M 128
#define A_MMI_M 111

void checkCUDAError(const char*);
void read_encrypted_file(int*);

/* Exercise 1.1 */
__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void affine_decrypt(int *d_input, int *d_output)
{
	/* Exercise 1.2 */
	int index = threadIdx.x;
	int value = d_input[index];
	value = modulo(A_MMI_M * (value - B), M);
	d_output[index] = value;
}

__global__ void affine_decrypt_multiblock(int *d_input, int *d_output)
{
	/* Exercise 1.8 */
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int value = d_input[index];
	value = modulo(A_MMI_M * (value - B), M);
	d_output[index] = value;
}


int main(int argc, char *argv[])
{
	int *h_input, *h_output;
	int *d_input, *d_output;
	unsigned int size;
	int i;
	cudaEvent_t start, stop; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);

	size = N * sizeof(int);

	/* allocate the host memory */
	h_input = (int *)malloc(size);
	h_output = (int *)malloc(size);

	/* Exercise 1.3: allocate device memory */
	cudaMalloc((void **)&d_input, size);
	cudaMalloc((void **)&d_output, size);
	checkCUDAError("Memory allocation");

	/* read the encryted text */
	read_encrypted_file(h_input);

	/* Exercise 1.4: copy host input to device input */
	cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	checkCUDAError("Input transfer to device");

	/* Exercise 1.5: Configure the grid of thread blocks and run the GPU kernel */
	// Single Block
	dim3 blocksPerGrid(1, 1, 1);
	dim3 threadsPerBlock(N, 1, 1);
	cudaEventRecord(start);
	affine_decrypt << < blocksPerGrid, threadsPerBlock >> > (d_input, d_output);
	cudaEventRecord(stop);

	/* wait for all threads to complete */
	cudaThreadSynchronize();
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	checkCUDAError("Kernel execution");


	//Multiple Block
	dim3 blocksPerGrid_M(8, 1, 1);
	dim3 threadsPerBlock_M(N / 8, 1, 1);
	cudaEventRecord(start);
	affine_decrypt_multiblock <<< blocksPerGrid_M, threadsPerBlock_M >>> (d_input, d_output);
	cudaEventRecord(stop);

	/* wait for all threads to complete */
	cudaThreadSynchronize();
	cudaEventSynchronize(stop); 
	float milliseconds_M = 0; 
	cudaEventElapsedTime(&milliseconds_M, start, stop);
	checkCUDAError("Kernel execution");
	
	/* Exercise 1.6: copy the gpu output back to the host */
	cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
	checkCUDAError("Result transfer to host");

	/* print out the result to screen */
	for (i = 0; i < N; i++) {
		printf("%c", (char)h_output[i]);
	}
	printf("\n");
	printf("==============================\n");
	printf("CUDA execution time took %.7f ms\n", milliseconds);
	printf("==============================\n");
	printf("==============================\n");
	printf("CUDA Multiple Block execution time took %.7f ms\n", milliseconds_M);
	printf("==============================\n");

	/* Exercise 1.7: free device memory */
	cudaFree(d_input);
	cudaFree(d_output);
	checkCUDAError("Free memory");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/* free host buffers */
	free(h_input);
	free(h_output);
	return 0;
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

void read_encrypted_file(int* input)
{
	FILE *f = NULL;
	f = fopen("encrypted01.bin", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find encrypted01.bin file \n");
		exit(1);
	}
	//read encrypted data
	fread(input, sizeof(unsigned int), N, f);
	fclose(f);
}