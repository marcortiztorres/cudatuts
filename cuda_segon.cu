#include <stdio.h>
#include <assert.h>
#include <cuda.h>

void incrementArrayOnHost(int *a, int N)
{
  int i;
  for (i=0; i < N; i++) a[i] = ++a[i];
}

__global__ void increment_array(int *a, int N){

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
 	if (idx<N) a[idx] = a[idx]+1.f;
}


int main(){

	int *a_d;
	int *a_h, *c_h;
	int i, N = 10;
	size_t size = N*sizeof(int);

	a_h = (float *)malloc(size);
	b_h = (float *)malloc(size);

	cudaMalloc(&a_d, size);
	for (i=0; i<N; i++) a_h[i] = (float)i;
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

	incrementArrayOnHost(a_h, N);

	int blockSize = 4;
	int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

	incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_d, N);

	cudaMemcpy(b_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
	for (i=0; i<N; i++) assert(a_h[i] == b_h[i]);
	free(a_h); free(b_h); cudaFree(a_d); 

}