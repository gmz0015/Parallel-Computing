#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#pragma warning(disable : 4996)

void checkCUDAError(const char*);


__device__ void modulo() {
	printf("%d: %d-%d-%d --- %d-%d-%d --- %d-%d-%d --- %d-%d-%d\n", str, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

__constant__ int d_c;
__device__ int d_value;

__global__ void affine_decrypt()
{
	modulo();
}

int main(int argc, char *argv[])
{
	int *h_value;
	h_value = (int *)malloc(sizeof(int));
	*h_value = 31;
	int i = 7;


	/* allocate the host memory */

	/* allocate device memory */
	//cudaMalloc((void **)&d, sizeof(int));
	checkCUDAError("Memory allocation");


	/* copy host input to device input */
	cudaMemcpyToSymbol(d_c, &i, sizeof(int));
	cudaMemcpyToSymbol(d_value, h_value, sizeof(int));
	checkCUDAError("Input transfer to device");

	/* Configure the grid of thread blocks and run the GPU kernel */
	dim3 blocksPerGrid(4, 3, 2);
	dim3 threadsPerBlock(1, 1, 1);
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