#define LENGTH_V 1024*1024
#define LENGTH_SHOW 10

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

	for (int offset = 1; offset < ndata; offset *= 2)
	{
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

void show_vector(char *myString, int lengthMyString, int *vector) {
	int j;
	printf("\n%s\n",myString);
	for (j = 0; j < lengthMyString; j++)
		printf("-");
	printf("\n");
	if (LENGTH_SHOW*2 < LENGTH_V) {
		for (j = 0; j < LENGTH_SHOW; j++)
			printf(" %d", vector[j]);
		printf(" ...");
		for (j = LENGTH_V-LENGTH_SHOW; j < LENGTH_V; j++)
			printf(" %d", vector[j]);
		printf("\n");
	} else {
		for (j=0 ; j<LENGTH_V; j++)
			printf(" %d", vector[j]);
		printf("\n");
	}

}

int main(void)
{
	int Vector[LENGTH_V], VectorScan[LENGTH_V];
	int j;
	clock_t start, end;
	double time_used;

	// -----------------------

	srand(time(NULL));
	for (j = 0; j < LENGTH_V; j++) {
		Vector[j] = (int)(rand() % 100) - 50;
	}

	char msg1[] = "Vector original";
	show_vector(msg1, strlen(msg1), Vector);

	// -----------------------
  // Vector scan (CPU)
  // -----------------------

	start = clock();

	VectorScan[0] = Vector[0];
	for (j = 1; j < LENGTH_V; j++) {
		VectorScan[j] = Vector[j] + VectorScan[j - 1];
	}

	end = clock();

	// -----------------------

	char msg2[] = "Vector scan (CPU)";
	show_vector(msg2, strlen(msg2), VectorScan);

	time_used = 1000.0*((double)(end-start)) / CLOCKS_PER_SEC;
	printf("CPU scan kernel processing time: %f millisec. (nº elements %d)\n",time_used, LENGTH_V);

}
