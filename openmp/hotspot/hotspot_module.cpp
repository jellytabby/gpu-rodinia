#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>

#include "hotspot_module.h"

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations
 * by one time step
 */
void single_iteration(FLOAT *result, FLOAT *temp, FLOAT *power, int row,
                      int col, FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1,
                      FLOAT step, FLOAT amb_temp, int num_omp_threads) {
  FLOAT delta;
  int r, c;
  int chunk;
  int num_chunk = row * col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
  int chunks_in_row = col / BLOCK_SIZE_C;
  int chunks_in_col = row / BLOCK_SIZE_R;

#ifdef OPEN
#ifndef __MIC__
  omp_set_num_threads(num_omp_threads);
#endif
#pragma omp parallel for shared(power, temp, result) private(chunk, r, c,      \
                                                                 delta)        \
    firstprivate(row, col, num_chunk, chunks_in_row) schedule(static)
#endif
  for (chunk = 0; chunk < num_chunk; ++chunk) {
    int r_start = BLOCK_SIZE_R * (chunk / chunks_in_col);
    int c_start = BLOCK_SIZE_C * (chunk % chunks_in_row);
    int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
    int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;

    if (r_start == 0 || c_start == 0 || r_end == row || c_end == col) {
      for (r = r_start; r < r_start + BLOCK_SIZE_R; ++r) {
        for (c = c_start; c < c_start + BLOCK_SIZE_C; ++c) {
          /* Corner 1 */
          if ((r == 0) && (c == 0)) {
            delta = (Cap_1) * (power[0] + (temp[1] - temp[0]) * Rx_1 +
                               (temp[col] - temp[0]) * Ry_1 +
                               (amb_temp - temp[0]) * Rz_1);
          } /* Corner 2 */
          else if ((r == 0) && (c == col - 1)) {
            delta = (Cap_1) * (power[c] + (temp[c - 1] - temp[c]) * Rx_1 +
                               (temp[c + col] - temp[c]) * Ry_1 +
                               (amb_temp - temp[c]) * Rz_1);
          } /* Corner 3 */
          else if ((r == row - 1) && (c == col - 1)) {
            delta = (Cap_1) *
                    (power[r * col + c] +
                     (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 +
                     (temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 +
                     (amb_temp - temp[r * col + c]) * Rz_1);
          } /* Corner 4	*/
          else if ((r == row - 1) && (c == 0)) {
            delta = (Cap_1) * (power[r * col] +
                               (temp[r * col + 1] - temp[r * col]) * Rx_1 +
                               (temp[(r - 1) * col] - temp[r * col]) * Ry_1 +
                               (amb_temp - temp[r * col]) * Rz_1);
          } /* Edge 1 */
          else if (r == 0) {
            delta =
                (Cap_1) *
                (power[c] + (temp[c + 1] + temp[c - 1] - 2.0 * temp[c]) * Rx_1 +
                 (temp[col + c] - temp[c]) * Ry_1 +
                 (amb_temp - temp[c]) * Rz_1);
          } /* Edge 2 */
          else if (c == col - 1) {
            delta =
                (Cap_1) * (power[r * col + c] +
                           (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] -
                            2.0 * temp[r * col + c]) *
                               Ry_1 +
                           (temp[r * col + c - 1] - temp[r * col + c]) * Rx_1 +
                           (amb_temp - temp[r * col + c]) * Rz_1);
          } /* Edge 3 */
          else if (r == row - 1) {
            delta = (Cap_1) *
                    (power[r * col + c] +
                     (temp[r * col + c + 1] + temp[r * col + c - 1] -
                      2.0 * temp[r * col + c]) *
                         Rx_1 +
                     (temp[(r - 1) * col + c] - temp[r * col + c]) * Ry_1 +
                     (amb_temp - temp[r * col + c]) * Rz_1);
          } /* Edge 4 */
          else if (c == 0) {
            delta = (Cap_1) * (power[r * col] +
                               (temp[(r + 1) * col] + temp[(r - 1) * col] -
                                2.0 * temp[r * col]) *
                                   Ry_1 +
                               (temp[r * col + 1] - temp[r * col]) * Rx_1 +
                               (amb_temp - temp[r * col]) * Rz_1);
          }
          result[r * col + c] = temp[r * col + c] + delta;
        }
      }
      continue;
    }

    for (r = r_start; r < r_start + BLOCK_SIZE_R; ++r) {
#pragma omp simd
      for (c = c_start; c < c_start + BLOCK_SIZE_C; ++c) {
        /* Update Temperatures */
        result[r * col + c] =
            temp[r * col + c] +
            (Cap_1 * (power[r * col + c] +
                      (temp[(r + 1) * col + c] + temp[(r - 1) * col + c] -
                       2.f * temp[r * col + c]) *
                          Ry_1 +
                      (temp[r * col + c + 1] + temp[r * col + c - 1] -
                       2.f * temp[r * col + c]) *
                          Rx_1 +
                      (amb_temp - temp[r * col + c]) * Rz_1));
      }
    }
  }
}

#ifdef OMP_OFFLOAD
#pragma offload_attribute(pop)
#endif

/* Transient solver driver routine: simply converts the heat
 * transfer differential equations to difference equations
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp,
                       FLOAT *power, int row, int col, FLOAT chip_height,
                       FLOAT chip_width, FLOAT t_chip, FLOAT amb_temp,
                       int omp_num_threads) {
#ifdef VERBOSE
  int i = 0;
#endif

  FLOAT grid_height = chip_height / row;
  FLOAT grid_width = chip_width / col;

  FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

  FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  FLOAT step = PRECISION / max_slope / 1000.0;

  FLOAT Rx_1 = 1.f / Rx;
  FLOAT Ry_1 = 1.f / Ry;
  FLOAT Rz_1 = 1.f / Rz;
  FLOAT Cap_1 = step / Cap;

#ifdef OMP_OFFLOAD
  int array_size = row * col;
#pragma omp target map(temp[0 : array_size])                                   \
    map(to : power[0 : array_size], row, col, Cap_1, Rx_1, Ry_1, Rz_1, step,   \
            num_iterations) map(result[0 : array_size])
#endif
  {
    FLOAT *r = result;
    FLOAT *t = temp;
    for (int i = 0; i < num_iterations; i++) {
      single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step,
                       amb_temp, omp_num_threads);
      FLOAT *tmp = t;
      t = r;
      r = tmp;
    }
  }
}
