#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define MAX_CHAR 100
#define DATAFILE "data256.txt"
#define RESULTSFILE "resultsSectionB256.txt"
#define G 6.674e-11
#define NUM_ITER 20000
#define NUM_ITER_SHOW 1000

__global__ void calculateAsteroid(double * d_x, double * d_y, double * d_vx, double * d_vy, double * d_m) {
  int indexInteractuateAsteroid = blockIdx.x * blockDim.x + threadIdx.x;
  int indexInitialAsteroid = blockIdx.y * blockDim.y + threadIdx.y;

  if (indexInteractuateAsteroid != indexInitialAsteroid) {
    double d = sqrt(pow((d_x[indexInitialAsteroid] - d_x[indexInteractuateAsteroid]), 2.0) + pow((d_y[indexInitialAsteroid] - d_y[indexInteractuateAsteroid]), 2.0));
    double f = G * d_m[indexInitialAsteroid] * d_m[indexInteractuateAsteroid] / pow(d, 2.0);
    double fx = f * (d_x[indexInteractuateAsteroid] - d_x[indexInitialAsteroid]) / d;
    double ax = fx / d_m[indexInitialAsteroid];
    double fy = f * (d_y[indexInteractuateAsteroid] - d_y[indexInitialAsteroid]) / d;
    double ay = fy / d_m[indexInitialAsteroid];
    atomicAdd(&d_vx[indexInitialAsteroid], ax);
    atomicAdd(&d_vy[indexInitialAsteroid], ay);
  }
}
__global__ void updatePositions(double * d_x, double * d_y, double * d_vx, double * d_vy) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  d_x[i] += d_vx[i];
  d_y[i] += d_vy[i];
}
int main(){

    clock_t start, end;
    double time_used;
    char  str[MAX_CHAR];
    FILE *file;
    int noOfObjects;
    int i;

    file = fopen( DATAFILE , "r");
    fscanf(file,"%s",str);
    noOfObjects = atoi(str);
    printf("Number of objects: %d\n",noOfObjects);

    const int OBJECTS_BYTES = noOfObjects * sizeof(double);

    double *h_x = (double *) malloc(sizeof(double) * noOfObjects);
    double *h_y = (double *) malloc(sizeof(double) * noOfObjects);
    double *h_vx = (double *) malloc(sizeof(double) * noOfObjects);
    double *h_vy = (double *) malloc(sizeof(double) * noOfObjects);
    double *h_m = (double *) malloc(sizeof(double) * noOfObjects);

    double *x0 = (double *) malloc(sizeof(double) * noOfObjects);
    double *y0 = (double *) malloc(sizeof(double) * noOfObjects);
    double *vx0 = (double *) malloc(sizeof(double) * noOfObjects);
    double *vy0 = (double *) malloc(sizeof(double) * noOfObjects);

    double *d_x, *d_y, *d_vx, *d_vy, *d_m;

    cudaMalloc((void**) &d_x, OBJECTS_BYTES);
    cudaMalloc((void**) &d_y, OBJECTS_BYTES);
    cudaMalloc((void**) &d_vx, OBJECTS_BYTES);
    cudaMalloc((void**) &d_vy, OBJECTS_BYTES);
    cudaMalloc((void**) &d_m, OBJECTS_BYTES);

    printf("\n");

    for (i = 0; i < noOfObjects; i++) {
      fscanf(file, "%s", str);
      h_x[i] = atof(str);
      x0[i] = atof(str);
      fscanf(file, "%s", str);
      h_y[i] = atof(str);
      y0[i] = atof(str);
      fscanf(file, "%s", str);
      h_vx[i] = atof(str);
      vx0[i] = atof(str);
      fscanf(file, "%s", str);
      h_vy[i] = atof(str);
      vy0[i] = atof(str);
      fscanf(file, "%s", str);
      h_m[i] = atof(str);
    }
    fclose(file);

    cudaMemcpy(d_x, h_x, OBJECTS_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, OBJECTS_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, OBJECTS_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, OBJECTS_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, OBJECTS_BYTES, cudaMemcpyHostToDevice);
    start = clock();
    for (int niter = 0;niter < NUM_ITER;niter++) {

      //1024
      //dim3 dimBlock(32, 32);
      //dim3 dimGrid(32, 32);
      //256
      dim3 dimBlock(32, 32);
      dim3 dimGrid(8, 8);
      calculateAsteroid<<<dimGrid, dimBlock>>>(d_x, d_y, d_vx, d_vy, d_m);
      updatePositions<<<1, noOfObjects>>>(d_x, d_y, d_vx, d_vy);

      if (niter % NUM_ITER_SHOW == 0) {
        printf("Iteration %d/%d\n", niter, NUM_ITER);
      }
    }  // nIter
    end = clock();
    cudaMemcpy(h_x, d_x, OBJECTS_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, OBJECTS_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vx, d_vx, OBJECTS_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vy, d_vy, OBJECTS_BYTES, cudaMemcpyDeviceToHost);
    file = fopen( RESULTSFILE , "w");
    fprintf(file, "Movement of objects\n");
    fprintf(file, "-------------------\n");
    for (i = 0; i < noOfObjects; i++) {
        double mov = sqrt(pow((x0[i] - h_x[i]), 2.0) + pow((y0[i] - h_y[i]), 2.0));
        fprintf(file,"  Object %i  -  %f meters\n", i, mov);
    }
    int hours = NUM_ITER / 3600;
    int mins = (NUM_ITER - hours * 3600) / 60;
    int secs = (NUM_ITER - hours * 3600 - mins * 60);
    fprintf(file,"Time elapsed: %i seconds (%i hours, %i minutes, %i seconds)\n", NUM_ITER, hours, mins, secs);

    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    fprintf(file,"Processing time: %f sec.\n", time_used);
    fclose(file);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_m);

    free(h_x);
    free(h_y);
    free(h_vx);
    free(h_vy);
    free(h_m);

    return 0;
}  // main