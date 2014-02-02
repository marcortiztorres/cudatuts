#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <iostream>

void checkCUDAError(const char *msg);

__global__ void product_two_vectors(int *d_a, int *d_b, int *d_r)
{
	int actual = blockIdx.x + threadIdx.x;
	d_r[actual] = d_a[actual] * d_b[actual];
}


int main(){

	int size_a = 100 * 1024;
	int *h_a;
	int *d_a, *d_b, *d_r;

	int numThreadsPerBlock = 100;
	int numBlocks = size_a / numThreadsPerBlock;

	malloc(size_a * sizeof(int));

	for (int i = 0; i < size_a; ++i) h_a[i] = i;
	cudaMalloc(&d_a, size_a * sizeof(int));
	cudaMalloc(&d_b, size_a * sizeof(int));
	cudaMalloc(&d_r, size_a * sizeof(int));
	checkCUDAError("Error creating space");

	cudaMemcpy(d_a, h_a, size_a * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("memcpy");
	for (int i = 0; i < size_a; ++i) h_a[i] = size_a - i - 1;

	cudaMemcpy(d_b, h_a, size_a * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("memcpy");

	dim3 blocks(numBlocks);
	dim3 threads(numThreadsPerBlock);

	sort_Array <<< blocks, threads >>> (d_a, d_b, d_r);
	cudaThreadSyncronize();

	cudamMemcpy(h_a, d_r, size_a * sizeof(int), cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy");

	for (int i = 0; i < count; ++i) 
	{
		assert (h_a[i] = i * (size_a - i));
		cout << h_a[i] << ",";
	}
	cudaFree(d_a);
	cudaFree(d_b);
	checkCUDAError("free_mem");
	free(h_a);

}

void checkCUDAError(const char *msg){

	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
	}

}
