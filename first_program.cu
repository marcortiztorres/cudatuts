__global__
void saveIDs(int *idsOut){
	const int tid = threadIdx.x;
	
	idsOut[tid] = tid;
}

#include <cstdlib>
#include <iostream>

int main(int argc, char **argv){
	const int numThreads = atoi(argv[1]);
	
	int *dIDs;
	
	cudaMalloc(&dIDs, sizeof(int) * numThreads);
	
	saveIDs <<< 1, numThreads >>>(dIDs);
	
	int *hIDs = new int [numThreads];

	cudaMemcpy (hIDs, dIDs, sizeof(int) * numThreads, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < numThreads; ++i){
		std::cout << i << ": " << hIDs[i] << std::endl;
	}
	
	delete[] hIDs;
	cudaFree(dIDs);
	
	return 0;
}
