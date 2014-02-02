#include <stdio.h>
#include <assert.h>
#include <cuda.h>

void checkCUDAError(const char *msg);

__global__ void reverseArray(int *a_b, int *b_d){
	int inOffset  = blockDim.x * blockIdx.x;
    int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
    int in  = inOffset + threadIdx.x;
    int out = outOffset + (blockDim.x - 1 - threadIdx.x);
    b_d[out] = a_b[in];
}


int main(){
	int *a_h;
	int *a_d, *b_d;

	int dimAB = 256 * 1024; //We want to create a 256KB array
	int numThreadsPerBlock = 256;
	int numBlocks = dimAB/numThreadsPerBlock;

	h_a = malloc(dimAB * sizeof(int));
	cudaMalloc(&a_d, dimAB * sizeof(int));
	cudaMalloc(&a_d, dimAB * sizeof(int));

	for (int i = 0; i < dimAB; ++i) a_h = i;

	cudaMemcpy(a_d, a_h, dimAB * sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    reverseArrayBlock<<< dimGrid, dimBlock >>>( b_d, a_d );

    cudaThreadSyncronize(); 

    checkCUDAError("kernel invocation");

    cudaMemcpy(a_h, a_d, dimAB * sizeof(int), cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    for (int i = 0; i < dimAB; ++i) assert (a_h[i] == dimAB - 1 - i);

    cudaFree(a_d);
	cudaFree(b_d);

	free(a_h);

	printf("Correct!\n");
 
    return 0;
}

void checkCUDAError(const char *msg){

	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
	}

}