#include <math.h>
#include <omp.h>
#include "backprop_module.h"
#define OPEN
#define ABS(x) (((x) > 0.0) ? (x) : (-(x)))

/*** The squashing function.  Currently, it's a sigmoid. ***/

float squash(float x) {
  float m;
  // x = -x;
  // m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  // return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x)));
}

void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2) {
  float sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
#pragma omp parallel for shared(conn, n1, n2, l1) private(k, j)                \
    reduction(+ : sum) schedule(static)
#endif
  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= n1; k++) {
      sum += conn[k][j] * l1[k];
    }
    l2[j] = squash(sum);
  }
}
void bpnn_output_error(float *delta, float *target, float *output, int nj,
                       float *err) {
  int j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
                       float **who, float *hidden, float *err) {
  int j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}

void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
                         float **w, float **oldw) {
  float new_dw;
  int k, j;
  ly[0] = 1.0;
  // eta = 0.3;
  // momentum = 0.3;

#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
#pragma omp parallel for shared(oldw, w, delta) private(j, k, new_dw)          \
    firstprivate(ndelta, nly)
#endif
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
}
