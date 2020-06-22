#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define DATAFILE "data.txt"
#define NUMBER_OF_ITEMS 1024
#define MIN_WEIGHT 500000
#define MAX_WEIGHT 10000000
#define MAX_DISTANCE 10000

double generateRandNumber(long minVal, long maxVal) {
    long r =random();
    double r2 = (double) r /RAND_MAX;
    return (double)minVal + r2*(maxVal-minVal);
}

int main() {
    srand(time(NULL));

    printf("Creating data of %d asteroids\n",NUMBER_OF_ITEMS);


    FILE *file;
    file = fopen( DATAFILE , "w");

    fprintf(file, "%d\n",NUMBER_OF_ITEMS);

    int i;
    for(i=0; i<NUMBER_OF_ITEMS; i++) {
        double x = generateRandNumber(-MAX_DISTANCE,MAX_DISTANCE);
        fprintf(file,"%.1f\n",x);
        double y = generateRandNumber(-MAX_DISTANCE,MAX_DISTANCE);
        fprintf(file,"%.1f\n",y);
        fprintf(file,"0.0\n");
        fprintf(file,"0.0\n");
        double m = generateRandNumber(MIN_WEIGHT,MAX_WEIGHT);
        fprintf(file,"%.1f\n",m);

    }

    fclose(file);
}