#include <stdio.h>
#include <iostream>
#include <cuda.h>
using namespace std;

void CheckCudaError(string &e);

__global__ void productMatrix(int *matrix_a, int *matrix_b, int *matrix_c)
{
	int suma;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < 100; ++i)
	{
		suma += matrix_a[row * 100 + i] * matrix_b[col * 100 + i];
	}
	matrix_c[(blockIdx.x * blockDim.x + threadIdx) * (blockIdx.y * blockDim.y + threadIdx.y)];
}

#define BLOCK_SIZE 5

int main(){

	//Creamos punteros para apuntar tanto al dispositivo como a memoria.
	int *h_a, *h_b;
	int *d_a, *d_b, *d_c;

	/*Trabajaremos con 20 blocks 
		es preferible blocks >= threads
		que blocks < threads
	*/

	int NumBlocks = 100 / BLOCK_SIZE;
	int num_elements = NumBlocks * BLOCK_SIZE;


	//Apuntamos los punteros hacia un espacio de 100*100 elementos en el host
	h_a = malloc(num_elements * sizeof(int));
	h_b = malloc(num_elements * sizeof(int));
	CheckCudaError("malloc_host_error");


	//LLenamos la memoria
	for (int i = 0; i < num_elements; ++i)
	{
		h_a[i] = i;
		h_b[i] = num_elements - 1 - i;
	}


	//Apuntamos los punteros del dispositivo hacia una reserva de memoria de 100*100 elementos.
	cudaMalloc(&d_a, num_elements * sizeof(int));
	cudaMalloc(&d_b, num_elements * sizeof(int));
	cudaMalloc(&d_c, num_elements * sizeof(int));
	CheckCudaError("malloc_device_error");


	/*Copiamos los elementos del host ya llenados anteriormente (llenamos memoria,
		copiando las matrizes del host hacia la tarjeta gráfica (device).*/
	cudaMemcpy(d_a, h_a, num_elements * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, num_elements * sizeof(int), cudaMemcpyHostToDevice);
	CheckCudaError("memcpy_error");


	free(h_b); 
	CheckCudaError("Free_host_error");


	//Establecemos el num de threas y blocks que utilizaremos
	dim3 gridDim (NumBlocks, NumBlocks);
	dim3 blockDim (BLOCK_SIZE, BLOCK_SIZE);
	//LLamamos la función.
	productMatrix <<< gridDim, blockDim >>> (d_a, d_b, d_c);
	CheckCudaError("Calling_device_function_error");


	/*Esperamos a que todos los threads hayan hecho su trabajo (multiplicar las matrizes)
		antes de copy back.*/
	cudaThreadSyncronize();
	CheckCudaError("Syncronize_threads_error");


	//Una vez sincronizados los volvemos a copiar hacia el host.
	cudaMemcpy(h_a, d_c, num_elements * sizeof(int), cudaMemcpyDeviceToHost);
	CheckCudaError("mempcy_host_error");


	//Imprimimos por pantalla
	for (int i = 0; i < num_elements; ++i) cout << h_a[i];


	//Aliberamos memoria en el device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	//Aliveramos meomria en host.
	free(h_a);

	CheckCudaError("free_device_error");

}


void CheckCudaError(string &e)
{	
	//Obtenemos el ultimo error.
	cudaError_t err = cudaGetLastError();
	//Si hay error imprime el error por pantalla
	if(cudaSuccess != err){
		cout << e << endl;
	}
}
