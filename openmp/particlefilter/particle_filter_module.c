#include <math.h>
#include <stdlib.h>
#include <limits.h>
#define PI 3.1415926535897932

/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
*/
long M = INT_MAX;
/**
@var A value for LCG
*/
int A = 1103515245;
/**
@var C value for LCG
*/
int C = 12345;
/**
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value
 * > input value
 */
double roundDouble(double value) {
  int newValue = (int)(value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}

/**
* Fills a radius x radius matrix representing the disk
* @param disk The pointer to the disk to be made
* @param radius  The radius of the disk to be made
*/
void strelDisk(int * disk, int radius)
{
	int diameter = radius*2 - 1;
	int x, y;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			double distance = sqrt(pow((double)(x-radius+1),2) + pow((double)(y-radius+1),2));
			if(distance < radius)
			disk[x*diameter + y] = 1;
		}
	}
}

/**
* Fills a 2D array describing the offsets of the disk object
* @param se The disk object
* @param numOnes The number of ones in the disk
* @param neighbors The array that will contain the offsets
* @param radius The radius used for dilation
*/
void getneighbors(int * se, int numOnes, double * neighbors, int radius){
	int x, y;
	int neighY = 0;
	int center = radius - 1;
	int diameter = radius*2 -1;
	for(x = 0; x < diameter; x++){
		for(y = 0; y < diameter; y++){
			if(se[x*diameter + y]){
				neighbors[neighY*2] = (int)(y - center);
				neighbors[neighY*2 + 1] = (int)(x - center);
				neighY++;
			}
		}
	}
}

/**
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
double randu(int * seed, int index)
{
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index]/((double) M));
}

/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
double randn(int * seed, int index){
	/*Box-Muller algorithm*/
	double u = randu(seed, index);
	double v = randu(seed, index);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}
/**
 * Finds the first element in the CDF that is greater than or equal to the
 * provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the
 * last index
 */
int findIndex(double *CDF, int lengthCDF, double value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1) {
    return lengthCDF - 1;
  }
  return index;
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In
 * addition, it references a provided MATLAB function which takes the video, the
 * objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */
void particleFilter(int *I, int IszX, int IszY, int Nfr, int *seed,
                    int Nparticles) {

  int max_size = IszX * IszY * Nfr;
  // long long start = get_time();
  // original particle centroid
  double xe = roundDouble(IszY / 2.0);
  double ye = roundDouble(IszX / 2.0);

  // expected object locations, compared to center
  int radius = 5;
  int diameter = radius * 2 - 1;
  int *disk = (int *)malloc(diameter * diameter * sizeof(int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  double *objxy = (double *)malloc(countOnes * 2 * sizeof(double));
  getneighbors(disk, countOnes, objxy, radius);

  // long long get_neighbors = get_time();
  // printf("TIME TO GET NEIGHBORS TOOK: %f\n", elapsed_time(start,
  // get_neighbors));
  // initial weights are all equal (1/Nparticles)
  double *weights = (double *)malloc(sizeof(double) * Nparticles);
#pragma omp parallel for shared(weights, Nparticles) private(x)
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((double)(Nparticles));
  }
  // long long get_weights = get_time();
  // printf("TIME TO GET WEIGHTSTOOK: %f\n", elapsed_time(get_neighbors,
  // get_weights));
  // initial likelihood to 0.0
  double *likelihood = (double *)malloc(sizeof(double) * Nparticles);
  double *arrayX = (double *)malloc(sizeof(double) * Nparticles);
  double *arrayY = (double *)malloc(sizeof(double) * Nparticles);
  double *xj = (double *)malloc(sizeof(double) * Nparticles);
  double *yj = (double *)malloc(sizeof(double) * Nparticles);
  double *CDF = (double *)malloc(sizeof(double) * Nparticles);
  double *u = (double *)malloc(sizeof(double) * Nparticles);
  int *ind = (int *)malloc(sizeof(int) * countOnes * Nparticles);
#pragma omp parallel for shared(arrayX, arrayY, xe, ye) private(x)
  for (x = 0; x < Nparticles; x++) {
    arrayX[x] = xe;
    arrayY[x] = ye;
  }
  int k;

  // printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights,
  // get_time()));
  int indX, indY;
  for (k = 1; k < Nfr; k++) {
// long long set_arrays = get_time();
// apply motion model
// draws sample from motion model (random walk). The only prior information
// is that the object moves 2x as fast as in the y direction
#pragma omp parallel for shared(arrayX, arrayY, Nparticles, seed) private(x)
    for (x = 0; x < Nparticles; x++) {
      arrayX[x] += 1 + 5 * randn(seed, x);
      arrayY[x] += -2 + 2 * randn(seed, x);
    }
// long long error = get_time();
// printf("TIME TO SET ERROR TOOK: %f\n", elapsed_time(set_arrays, error));
// particle filter likelihood
#pragma omp parallel for shared(likelihood, I, arrayX, arrayY, objxy,          \
                                    ind) private(x, y, indX, indY)
    for (x = 0; x < Nparticles; x++) {
      // compute the likelihood: remember our assumption is that you know
      //  foreground and the background image intensity distribution.
      //  Notice that we consider here a likelihood ratio, instead of
      //  p(z|x). It is possible in this case. why? a hometask for you.
      // calc ind
      for (y = 0; y < countOnes; y++) {
        indX = roundDouble(arrayX[x]) + objxy[y * 2 + 1];
        indY = roundDouble(arrayY[x]) + objxy[y * 2];
        ind[x * countOnes + y] = fabs(indX * IszY * Nfr + indY * Nfr + k);
        if (ind[x * countOnes + y] >= max_size)
          ind[x * countOnes + y] = 0;
      }
      likelihood[x] = 0;
      for (y = 0; y < countOnes; y++)
        likelihood[x] += (pow((I[ind[x * countOnes + y]] - 100), 2) -
                          pow((I[ind[x * countOnes + y]] - 228), 2)) /
                         50.0;
      likelihood[x] = likelihood[x] / ((double)countOnes);
    }
// long long likelihood_time = get_time();
// printf("TIME TO GET LIKELIHOODS TOOK: %f\n", elapsed_time(error,
// likelihood_time)); update & normalize weights using equation (63) of
// Arulampalam Tutorial
#pragma omp parallel for shared(Nparticles, weights, likelihood) private(x)
    for (x = 0; x < Nparticles; x++) {
      weights[x] = weights[x] * exp(likelihood[x]);
    }
    // long long exponential = get_time();
    // printf("TIME TO GET EXP TOOK: %f\n", elapsed_time(likelihood_time,
    // exponential));
    double sumWeights = 0;
#pragma omp parallel for private(x) reduction(+ : sumWeights)
    for (x = 0; x < Nparticles; x++) {
      sumWeights += weights[x];
    }
// long long sum_time = get_time();
// printf("TIME TO SUM WEIGHTS TOOK: %f\n", elapsed_time(exponential,
// sum_time));
#pragma omp parallel for shared(sumWeights, weights) private(x)
    for (x = 0; x < Nparticles; x++) {
      weights[x] = weights[x] / sumWeights;
    }
    // long long normalize = get_time();
    // printf("TIME TO NORMALIZE WEIGHTS TOOK: %f\n", elapsed_time(sum_time,
    // normalize));
    xe = 0;
    ye = 0;
// estimate the object location by expected values
#pragma omp parallel for private(x) reduction(+ : xe, ye)
    for (x = 0; x < Nparticles; x++) {
      xe += arrayX[x] * weights[x];
      ye += arrayY[x] * weights[x];
    }
    // long long move_time = get_time();
    // printf("TIME TO MOVE OBJECT TOOK: %f\n", elapsed_time(normalize,
    // move_time)); printf("XE: %lf\n", xe); printf("YE: %lf\n", ye);
    double distance = sqrt(pow((double)(xe - (int)roundDouble(IszY / 2.0)), 2) +
                           pow((double)(ye - (int)roundDouble(IszX / 2.0)), 2));
    // printf("%lf\n", distance);
    // //display(hold off for now)

    // pause(hold off for now)

    // resampling

    CDF[0] = weights[0];
    for (x = 1; x < Nparticles; x++) {
      CDF[x] = weights[x] + CDF[x - 1];
    }
    // long long cum_sum = get_time();
    // printf("TIME TO CALC CUM SUM TOOK: %f\n", elapsed_time(move_time,
    // cum_sum));
    double u1 = (1 / ((double)(Nparticles))) * randu(seed, 0);
#pragma omp parallel for shared(u, u1, Nparticles) private(x)
    for (x = 0; x < Nparticles; x++) {
      u[x] = u1 + x / ((double)(Nparticles));
    }
    // long long u_time = get_time();
    // printf("TIME TO CALC U TOOK: %f\n", elapsed_time(cum_sum, u_time));
    int j, i;

#pragma omp parallel for shared(CDF, Nparticles, xj, yj, u, arrayX,            \
                                    arrayY) private(i, j)
    for (j = 0; j < Nparticles; j++) {
      i = findIndex(CDF, Nparticles, u[j]);
      if (i == -1)
        i = Nparticles - 1;
      xj[j] = arrayX[i];
      yj[j] = arrayY[i];
    }
    // long long xyj_time = get_time();
    // printf("TIME TO CALC NEW ARRAY X AND Y TOOK: %f\n", elapsed_time(u_time,
    // xyj_time));

    // #pragma omp parallel for shared(weights, Nparticles) private(x)
    for (x = 0; x < Nparticles; x++) {
      // reassign arrayX and arrayY
      arrayX[x] = xj[x];
      arrayY[x] = yj[x];
      weights[x] = 1 / ((double)(Nparticles));
    }
    // long long reset = get_time();
    // printf("TIME TO RESET WEIGHTS TOOK: %f\n", elapsed_time(xyj_time,
    // reset));
  }
  free(disk);
  free(objxy);
  free(weights);
  free(likelihood);
  free(xj);
  free(yj);
  free(arrayX);
  free(arrayY);
  free(CDF);
  free(u);
  free(ind);
}
