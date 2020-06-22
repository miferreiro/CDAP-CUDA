#define LENGTH_V 1024*1024
#define LENGTH_SHOW 10

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <time.h>

void show_vector(char *myString, int lengthMyString, thrust::host_vector<int> vector) {
	int j;
  printf("\n%s\n",myString);
  for (j = 0; j < lengthMyString; j++)
    printf("-");
  printf("\n");
  if (LENGTH_SHOW * 2 < LENGTH_V) {
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
  printf("\n");
}

int RandomNumber () { return ( (std::rand() % 100) - 50); }

int main(void) {

  clock_t start, end;
  double time_used;

  // -----------------------

  thrust::host_vector<int> h_vec(LENGTH_V);
  srand(time(NULL));
  std::generate(h_vec.begin(), h_vec.end(), RandomNumber);

	char msg1[] = "Vector original";
	show_vector(msg1, strlen(msg1), h_vec);

  // -----------------------
  // Vector scan (Thrust)
  // -----------------------

  thrust::device_vector<int> d_vec = h_vec;

  start = clock();

  thrust::inclusive_scan(d_vec.begin(), d_vec.end(), d_vec.begin());

  end = clock();

  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  // -----------------------

	char msg2[] = "Scan result (GPU)";
	show_vector(msg2, strlen(msg2), h_vec);

  time_used = 1000.0 * ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Thrust scan kernel processing time: %f millisec. (nº elements %d)\n",time_used, LENGTH_V);
  printf("...\n");

  return 0;
}