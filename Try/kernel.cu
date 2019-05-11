#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#pragma warning(disable : 4996)

void checkCUDAError(const char*);


__device__ int modulo(int a, int b) {
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__device__ int d_value;

__global__ void affine_decrypt()
{
	//int d_value;
	int *temp = &d_value;
	///d_value = *temp + threadIdx.x;
	//d_value = temp
	atomicAdd(&temp[0], 1);
	printf("%d-%d: %d\n", blockIdx.x, threadIdx.x, d_value);
}

int main(int argc, char *argv[])
{
	int *h_value = 31;
	h_value = (int *)malloc(sizeof(int));
	*h_value = 31;
	int i = 7;


	/* allocate the host memory */

	/* allocate device memory */
	//cudaMalloc((void **)&d, sizeof(int));
	checkCUDAError("Memory allocation");


	/* copy host input to device input */
	cudaMemcpyToSymbol(d_value, h_value, sizeof(int));
	checkCUDAError("Input transfer to device");

	/* Configure the grid of thread blocks and run the GPU kernel */
	dim3 blocksPerGrid(16, 1, 1);
	dim3 threadsPerBlock(8, 1, 1);
	affine_decrypt << <blocksPerGrid, threadsPerBlock >> > ();

	/* wait for all threads to complete */
	cudaThreadSynchronize();
	checkCUDAError("Kernel execution");

	/* copy the gpu output back to the host */
	cudaMemcpyFromSymbol(h_value, d_value, sizeof(int));
	checkCUDAError("Result transfer to host");

	/* print out the result to screen */
	printf("Result: %d\n", *h_value);

	/* free device memory */
	//cudaFree(d_value);
	checkCUDAError("Free memory");

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