#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>

using namespace std;


void checkCudaError(string &s){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		cout << "Error" << s << endl;
	}
}
 
 __global__ void reverseArray(int *d_in, int *d_out)
 {

 	extern __shared__ int s_data[];

 	int posIn = blockDim * blockIdx.x + threadIdx.x;
 	s_data[blockDim - 1 - threadIdx.x] = d_in[posIn];

 	__syncthreads(); //Thanks GOD

 	int posOut = blockDim.x * (gridDim.x - 1 - blockIdx.x) + threadIdx.x;
    	d_out[posOut] = s_data[threadIdx.x];

 }



 int main(){

 	int *h_a, *d_a, *d_b;
 	int num_elem = 256 * 1000;

 	h_a = malloc(num_elem * sizeof(int));

 	for (int i = 0; i < num_elem; ++i)
 	{
 		h_a[i] = i;
 	}

 	int numThreadsPerBlock = 256;
 	int numBlocks = num_elem / numThreadsPerBlock;
 	int sharedMemSize = numThreadsPerBlock * sizeof(int);

 	cudaMalloc(&d_a, num_elem * sizeof(int));
 	cudaMalloc(&d_b, num_elem * sizeof(int));
 	checkCudaError("malloc");

 	cudaMemcpy(d_a, h_a, num_elem * sizeof(int), cudaMemcpyHostToDevice);
 	checkCudaError("memcpy");

 	dim3 blockDim(numThreadsPerBlock);
 	dim3 gridDim(numBlocks);
 	reverseArray <<< gridDim, blockDim, sharedMemSize >>> (d_a, d_b);

 	cudaThreadSyncronize();

 	cudaMemcpy(h_a, d_b, num_elem * sizeof(int), cudaMemcpyDeviceToHost);
 	checkCudaError("mempcy"); 

 	for (int i = 0; i < num_elem; ++i) 
 	{
 		assert (h_a[i] == num_elem - 1 - i);
 		cout << 
 	}

 	cudaFree(d_a);
 	cudaFree(d_b);
 	free(h_a);

 }
