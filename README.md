# Starting on CUDA

These three exercises were made in the subject of "Computación Distribuída e de Altas Prestacións" in the Master Degree of Computer Engineering of the University of Vigo in 2020

### Introduction to CUDA

Implement a program to do the next operations:
1. Initialize a square matrix of 10.000 elements containing random values.
2. Invoke a kernel with a grid composed by block of 16x16 threads.
3. Each thread must square one element of the matrix and store it in a result matrix.
4. The program will show the original matrix and the result matrix (or part of them) on the screen.

### Asteroids field on CUDA

The exercise consists of simulating an asteroid field that interacts two by two by means of gravitational attraction. To generate data to work with, GenerateAsteroids.c program is provided that will create a data.txt file with randomly generated asteroid data. With the statement attach two files data256.txt and data1024.txt with data of 256 and 1024 asteroids, respectively, located in a 20 km-side bi-dimensional area with masses between 500 and 10 million tons.

This exercise is divided into 3 sections:

#### Section A

Implementation of asteroid field simulation using GPU computing resources is proposed. In this first version, a kernel composed of a single one-dimensional block of N threads (N is the number of asteroids) will be used.

Each thread of the kernel will take care of one asteroid. It is recommended to implement the main loop of the simulation outside the kernel, that is, each call to the kernel will correspond to an iteration.

#### Section B

To increase the level of parallelism, a kernel will be implemented in which each thread will calculate the interaction between two asteroids. Each thread calculate the acceleration variations on the x and y axis and accumulate them in the speeds. Comments concerning this implementation:

1. It is not possible to perform the calculation using a kernel with only one block, because exceeds the maximum thread count per GPU block. For 256 recommended to use a two-dimensional grid of 8x8 blocks, each one of which consists of 32x32 threads. In the case of 1024 asteroids, it is recommended to use a two-dimensional grid of 32x32 blocks each of which shall be composed of 32x32 threads.

2. Each thread will calculate the accelerations ax and ay, and will accumulate them in the velocities vx and vy of the asteroids. To do this, it is necessary to use an atomic operation (atomicAdd) that is implemented for double variables on GPUs with a computing capability greater than 6.0 (that of the practice team is 6.1).

The implementation thus carried out gives a running time in the  practices of about 11 seconds (256 asteroids) or 47 seconds (1024  asteroids).

#### Section C

In the implementation of the previous section, each thread is in charge of calculating an interaction, both in the x-axis and in the y-axis. The program can be further optimized by increasing the degree of parallelism.

It is proposed to implement a kernel in which each block is three-dimensional, with a dimension equal to 2 in the z axis. The threads with z=0 will take care of the projections of the interactions in the x-axis, while the threads with z=1 will take care of the projections in the y-axis. If blocks of 32x32 threads are used, the maximum number of threads per block will be reached, so there will be two possibilities: (1) reduce the size of the blocks, or (2) use two-dimensional with a three-dimensional grid.

### SCAN-based algorithm on CUDA

An algorithm is implemented on Cuda to apply some kind of process to a large vector. Using the SCAN algorithm, the processes can be performed in parallel. If the vector is large, the implementation will have to be divided into blocks.

Once the algorithm is implemented in blocks, compare the time it takes to process a large vector in series and using the scan algorithm on GPUs.

In addition, the same program is implemented using the thrust library.

NOTE: Due to hardware limitations, the blocks will be one dimensional and 1024 thread size.