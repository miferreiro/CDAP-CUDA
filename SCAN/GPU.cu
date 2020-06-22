#define LENGTH_V  1024*1024
#define LENGTH_SHOW 10
#define BLOCK_SIZE 1024

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

// Kernel: naive scan
__global__ void scan_incl(int *g_odata, int *g_idata, int ndata) {
	int thid = threadIdx.x;
	extern __shared__ int temp[];
		// Double buffer dinamically reserved

	int pout = 0, pin = 1;
	temp[pout * ndata + thid] = g_idata[thid];
	__syncthreads();

	for (int offset = 1; offset < ndata; offset *= 2) {
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;

		temp[pout * ndata + thid] = temp[pin * ndata + thid];
		if (thid >= offset) {
			temp[pout * ndata + thid] += temp[pin * ndata + thid - offset];
		}
		__syncthreads();
	}
  g_odata[thid] = temp[pout * ndata + thid]; // write output

}
// Kernel: block scan
__global__ void scan_incl_blocks(int *g_odata, int *g_idata, int blockSize, int ndata) {

  int block = blockIdx.x;
  int thid = threadIdx.x;
  int posArray = block * blockSize + thid;
  int numElemts;

  if (block == (ceil( (double) ndata / (double) blockSize) - 1) &&  ndata % blockSize != 0) {
    numElemts = ndata % blockSize;
  } else {
    numElemts = blockSize;
  }

	extern __shared__ int temp[];
  // Double buffer dinamically reserved
  int pout = 0, pin = 1;
  temp[pout * numElemts + thid] = g_idata[posArray];
  __syncthreads();

  for (int offset = 1; offset < numElemts; offset *= 2) {
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;

		temp[pout * numElemts + thid] = temp[pin * numElemts + thid];
		if (thid >= offset) {
			temp[pout * numElemts + thid] += temp[pin * numElemts + thid - offset];
    }
		__syncthreads();
  }

  g_odata[posArray] = temp[pout * numElemts + thid];
}

__global__ void scan_getBlocksAux(int *g_block, int *g_idata, int blockSize, int ndata) {

  int block = blockIdx.x;
  int thid = threadIdx.x;
  int posArray = block * blockSize + thid;

  int numElemts;

  if (block == (ceil( (double) ndata / (double) blockSize) - 1) &&  ndata % blockSize != 0) {
    numElemts = ndata % blockSize;
  } else {
    numElemts = blockSize;
  }

  if (thid == (numElemts - 1)) {
    g_block[block] = g_idata[posArray];
  }
}

__global__ void addToBlocks(int *g_odata, int *g_block, int blockSize) {

  int block = blockIdx.x;
  int thid = threadIdx.x;
  int posArray = block * blockSize + thid;

  if (block != 0) {
    g_odata[posArray] += g_block[block - 1];
  }
}

void show_vector(char *myString, int lengthMyString, int *vector, int length) {
	int j;
  printf("\n%s\n",myString);
  for (j = 0; j < lengthMyString; j++)
    printf("-");
  printf("\n");
  if (LENGTH_SHOW * 2 < length) {
		for (j = 0; j < LENGTH_SHOW; j++)
			printf(" %d", vector[j]);
		printf(" ...");
		for (j = length-LENGTH_SHOW; j < length; j++)
			printf(" %d", vector[j]);
		printf("\n");
	} else {
		for (j=0 ; j<length; j++)
			printf(" %d", vector[j]);
		printf("\n");
	}
}

int main(void) {

	int h_VectorOriginal[LENGTH_V];
  int h_VectorScanGPU[LENGTH_V];

  int *d_VectorOriginal, *d_VectorScanGPUBlocks, *d_VectorScanGPUBlocksAux, *d_VectorScanGPUBlocks2;

	int j;
	clock_t start, end;
	double time_used;

  // -----------------------

  srand(time(NULL));
	for (j = 0; j < LENGTH_V; j++) {
    h_VectorOriginal[j] = (int)(rand() % 100) - 50;
  }

	char msg1[] = "Vector original";
	show_vector(msg1, strlen(msg1), h_VectorOriginal, LENGTH_V);

  // -----------------------
  // Vector scan blocks (GPU)
  // -----------------------

  cudaMalloc((void **)&d_VectorOriginal, sizeof(int) * LENGTH_V);
  cudaMalloc((void **)&d_VectorScanGPUBlocks, sizeof(int) * LENGTH_V);

  cudaMemcpy(d_VectorOriginal, h_VectorOriginal, sizeof(int) * LENGTH_V, cudaMemcpyHostToDevice);

  int numBlocks = ceil( (double) LENGTH_V / (double) BLOCK_SIZE);

  printf("\nNumero de bloques: %d\n", numBlocks);

  int numThreads = BLOCK_SIZE;
  printf("Numero de threads: %d\n", numThreads);

  cudaMalloc((void **)&d_VectorScanGPUBlocksAux, sizeof(int) * numBlocks);
  cudaMalloc((void **)&d_VectorScanGPUBlocks2, sizeof(int) * numBlocks);

  start = clock();

    scan_incl_blocks <<< numBlocks, numThreads, BLOCK_SIZE * 2 * sizeof(int) >>>(
      d_VectorScanGPUBlocks,
      d_VectorOriginal,
      BLOCK_SIZE,
      LENGTH_V);

    scan_getBlocksAux <<< numBlocks, numThreads >>>(
      d_VectorScanGPUBlocksAux,
      d_VectorScanGPUBlocks,
      BLOCK_SIZE,
      LENGTH_V
    );

    scan_incl <<< 1, numBlocks, numBlocks * 2 * sizeof(int) >>>(
      d_VectorScanGPUBlocks2,
      d_VectorScanGPUBlocksAux,
      numBlocks
    );

    addToBlocks <<< numBlocks, numThreads >>>(
      d_VectorScanGPUBlocks,
      d_VectorScanGPUBlocks2,
      BLOCK_SIZE
    );

  end = clock();

	cudaMemcpy(h_VectorScanGPU, d_VectorScanGPUBlocks, sizeof(int) * LENGTH_V, cudaMemcpyDeviceToHost);

  // -----------------------

	char msg3[] = "Vector scan blocks (GPU)";
	show_vector(msg3, strlen(msg3), h_VectorScanGPU, LENGTH_V);

  time_used = 1000.0 * ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("GPU scan kernel processing time: %f millisec. (nº elements %d)\n",time_used, LENGTH_V);

	cudaFree(d_VectorOriginal);
  cudaFree(d_VectorScanGPUBlocks);
  cudaFree(d_VectorScanGPUBlocksAux);
}