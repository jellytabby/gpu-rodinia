/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP
 */
#include "particle_filter_module.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

void __mc_profiling_begin(void);
void __mc_profiling_end(void);
/*****************************
 *GET_TIME
 *returns a long int representing the time
 *****************************/
long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times
float elapsed_time(long long start_time, long long end_time) {
  return (float)(end_time - start_time) / (1000 * 1000);
}
/**
 * Set values of the 3D array to a newValue if that value is equal to the
 * testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void setIf(int testValue, int newValue, int *array3D, int *dimX, int *dimY,
           int *dimZ) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}
/**
 * Sets values of 3D matrix using randomly generated numbers from a normal
 * distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(int *array3D, int *dimX, int *dimY, int *dimZ, int *seed) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] =
            array3D[x * *dimY * *dimZ + y * *dimZ + z] +
            (int)(5 * randn(seed, 0));
      }
    }
  }
}
/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
void dilate_matrix(int *matrix, int posX, int posY, int posZ, int dimX,
                   int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0)
    startX++;
  int startY = posY - error;
  while (startY < 0)
    startY++;
  int endX = posX + error;
  while (endX > dimX)
    endX--;
  int endY = posY + error;
  while (endY > dimY)
    endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      double distance =
          sqrt(pow((double)(x - posX), 2) + pow((double)(y - posY), 2));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
void imdilate_disk(int *matrix, int dimX, int dimY, int dimZ, int error,
                   int *newMatrix) {
  int x, y, z;
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}
/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the backgrounf intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
void videoSequence(int *I, int IszX, int IszY, int Nfr, int *seed) {
  int k;
  int max_size = IszX * IszY * Nfr;
  /*get object centers*/
  int x0 = (int)roundDouble(IszY / 2.0);
  int y0 = (int)roundDouble(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  /*move point*/
  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  /*dilate matrix*/
  int *newMatrix = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] =
            newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);

  /*define background, add noise*/
  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  /*add noise*/
  addNoise(I, &IszX, &IszY, &Nfr, seed);
}
/**
 * Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 -
 * (IK[IND] - 228)^2)/ 100
 * @param I The 3D matrix
 * @param ind The current ind array
 * @param numOnes The length of ind array
 * @return A double representing the sum
 */
double calcLikelihoodSum(int *I, int *ind, int numOnes) {
  double likelihoodSum = 0.0;
  int y;
  for (y = 0; y < numOnes; y++)
    likelihoodSum +=
        (pow((I[ind[y]] - 100), 2) - pow((I[ind[y]] - 228), 2)) / 50.0;
  return likelihoodSum;
}
/**
 * Finds the first element in the CDF that is greater than or equal to the
 * provided value and returns that index
 * @note This function uses binary search before switching to sequential search
 * @param CDF The CDF
 * @param beginIndex The index to start searching from
 * @param endIndex The index to stop searching
 * @param value The value to find
 * @return The index of value in the CDF; if value is never found, returns the
 * last index
 * @warning Use at your own risk; not fully tested
 */
int findIndexBin(double *CDF, int beginIndex, int endIndex, double value) {
  if (endIndex < beginIndex)
    return -1;
  int middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
  /*check the value*/
  if (CDF[middleIndex] >= value) {
    /*check that it's good*/
    if (middleIndex == 0)
      return middleIndex;
    else if (CDF[middleIndex - 1] < value)
      return middleIndex;
    else if (CDF[middleIndex - 1] == value) {
      while (middleIndex > 0 && CDF[middleIndex - 1] == value)
        middleIndex--;
      return middleIndex;
    }
  }
  if (CDF[middleIndex] > value)
    return findIndexBin(CDF, beginIndex, middleIndex + 1, value);
  return findIndexBin(CDF, middleIndex - 1, endIndex, value);
}
int main(int argc, char *argv[]) {

  char *usage = "openmp.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
  // check number of arguments
  if (argc != 9) {
    printf("%s\n", usage);
    return 0;
  }
  // check args deliminators
  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") ||
      strcmp(argv[7], "-np")) {
    printf("%s\n", usage);
    return 0;
  }

  int IszX, IszY, Nfr, Nparticles;

  // converting a string to a integer
  if (sscanf(argv[2], "%d", &IszX) == EOF) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[4], "%d", &IszY) == EOF) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[6], "%d", &Nfr) == EOF) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  // converting a string to a integer
  if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }

  if (Nparticles <= 0) {
    printf("Number of particles must be > 0\n");
    return 0;
  }
  // establish seed
  int *seed = (int *)malloc(sizeof(int) * Nparticles);
  int i;
  for (i = 0; i < Nparticles; i++)
    seed[i] = time(0) * i;
  // malloc matrix
  int *I = (int *)malloc(sizeof(int) * IszX * IszY * Nfr);
  long long start = get_time();
  // call video sequence
  videoSequence(I, IszX, IszY, Nfr, seed);
  long long endVideoSequence = get_time();
  printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
  // call particle filter
  __mc_profiling_begin();
  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
  __mc_profiling_end();
  long long endParticleFilter = get_time();
  printf("PARTICLE FILTER TOOK %f\n",
         elapsed_time(endVideoSequence, endParticleFilter));
  printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));

  free(seed);
  free(I);
  return 0;
}
