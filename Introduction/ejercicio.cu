#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 100
#define BLOCK_DIM 16

__global__ void matrixSquared(int *initialMatrix, int *finalMatrix) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int index = col + row * MATRIX_SIZE;

  if (col < MATRIX_SIZE && row < MATRIX_SIZE) {
    finalMatrix[index] = initialMatrix[index] * initialMatrix[index];
  }
}

int main(int argc, char ** argv) {

	const int MATRIX_BYTES = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);

	// generate the input matrix on the host
  int h_in[MATRIX_SIZE][MATRIX_SIZE];
  printf("Initial matrix\n");
  int i, j;

  for (i = 0; i < MATRIX_SIZE;i++) {
    for (j = 0; j < MATRIX_SIZE;j++) {
      h_in[i][j] = rand() % 10;
      printf("%d ", h_in[i][j]);
    }
    printf("\n");
  }

  int h_out[MATRIX_SIZE][MATRIX_SIZE];

  // declare GPU memory pointers
  int * d_in;
  int * d_out;

  // allocate GPU memory
  cudaMalloc((void**) &d_in, MATRIX_BYTES);
  cudaMalloc((void**) &d_out, MATRIX_BYTES);

  // transfer the matrix to the GPU
  cudaMemcpy(d_in, h_in, MATRIX_BYTES, cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
  int dimX = (MATRIX_SIZE + dimBlock.x - 1) / dimBlock.x;
  int dimY = (MATRIX_SIZE + dimBlock.y - 1) / dimBlock.y;
  dim3 dimGrid(dimX, dimY);
  //printf("dimGrid.x = %d, dimGrid.y = %d\n", dimGrid.x, dimGrid.y);

  // launch the kernel
	matrixSquared<<<dimGrid, dimBlock>>>(d_in, d_out);

  // copy back the result matrix to the CPU
  cudaMemcpy(h_out, d_out, MATRIX_BYTES, cudaMemcpyDeviceToHost);

  // print out the resulting matrix
  printf("Result matrix\n");
  for (i = 0; i < MATRIX_SIZE;i++) {
    for (j = 0; j < MATRIX_SIZE;j++) {
      printf("%d ", h_out[i][j]);
    }
    printf("\n");
  }

  cudaFree(d_in);
	cudaFree(d_out);

  return 0;
}