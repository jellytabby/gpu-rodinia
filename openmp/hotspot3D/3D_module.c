#include <omp.h>
void computeTempOMP(float *pIn, float *tIn, float *tOut, int nx, int ny, int nz,
                    float Cap, float Rx, float Ry, float Rz, float dt,
                    int numiter, float amb_temp) {

  float ce, cw, cn, cs, ct, cb, cc;

  float stepDivCap = dt / Cap;
  ce = cw = stepDivCap / Rx;
  cn = cs = stepDivCap / Ry;
  ct = cb = stepDivCap / Rz;

  cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

#pragma omp parallel
  {
    int count = 0;
    float *tIn_t = tIn;
    float *tOut_t = tOut;

    do {
      int z;
#pragma omp for
      for (z = 0; z < nz; z++) {
        int y;
        for (y = 0; y < ny; y++) {
          int x;
          for (x = 0; x < nx; x++) {
            int c, w, e, n, s, b, t;
            c = x + y * nx + z * nx * ny;
            w = (x == 0) ? c : c - 1;
            e = (x == nx - 1) ? c : c + 1;
            n = (y == 0) ? c : c - nx;
            s = (y == ny - 1) ? c : c + nx;
            b = (z == 0) ? c : c - nx * ny;
            t = (z == nz - 1) ? c : c + nx * ny;
            tOut_t[c] = cc * tIn_t[c] + cw * tIn_t[w] + ce * tIn_t[e] +
                        cs * tIn_t[s] + cn * tIn_t[n] + cb * tIn_t[b] +
                        ct * tIn_t[t] + (dt / Cap) * pIn[c] + ct * amb_temp;
          }
        }
      }
      float *t = tIn_t;
      tIn_t = tOut_t;
      tOut_t = t;
      count++;
    } while (count < numiter);
  }
  return;
}
