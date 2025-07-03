#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "hotspot_module.h"

extern "C" void __mc_profiling_begin(void);
extern "C" void __mc_profiling_end(void);

// Returns the current system time in microseconds
long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}
using namespace std;

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5
#define OPEN
// #define NUM_THREAD 4

typedef float FLOAT;

/* chip parameters	*/
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;

#ifdef OMP_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

/* ambient temperature, assuming no package at all	*/
const FLOAT amb_temp = 80.0;

int num_omp_threads;

void fatal(char *s) {
  fprintf(stderr, "error: %s\n", s);
  exit(1);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, char *file) {

  int i, j, index = 0;
  FILE *fp;
  char str[STR_SIZE];

  if ((fp = fopen(file, "w")) == 0)
    printf("The file was not opened\n");

  for (i = 0; i < grid_rows; i++)
    for (j = 0; j < grid_cols; j++) {

      sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
      fputs(str, fp);
      index++;
    }

  fclose(fp);
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file) {
  int i, index;
  FILE *fp;
  char str[STR_SIZE];
  FLOAT val;

  fp = fopen(file, "r");
  if (!fp)
    fatal("file could not be opened for reading");

  for (i = 0; i < grid_rows * grid_cols; i++) {
    fgets(str, STR_SIZE, fp);
    if (feof(fp))
      fatal("not enough lines in file");
    if ((sscanf(str, "%f", &val) != 1))
      fatal("invalid file format");
    vect[i] = val;
  }

  fclose(fp);
}

void usage(int argc, char **argv) {
  fprintf(stderr,
          "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of "
          "threads><temp_file> <power_file>\n",
          argv[0]);
  fprintf(stderr,
          "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
  fprintf(
      stderr,
      "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<no. of threads>   - number of threads\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial "
                  "temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated "
                  "power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file\n");
  exit(1);
}

int main(int argc, char **argv) {
  int grid_rows, grid_cols, sim_time, i;
  FLOAT *temp, *power, *result;
  char *tfile, *pfile, *ofile;

  /* check validity of inputs	*/
  if (argc != 8)
    usage(argc, argv);
  if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 ||
      (sim_time = atoi(argv[3])) <= 0 || (num_omp_threads = atoi(argv[4])) <= 0)
    usage(argc, argv);

  /* allocate memory for the temperature and power arrays	*/
  temp = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
  power = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
  result = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
  if (!temp || !power)
    fatal("unable to allocate memory");

  /* read initial temperatures and input power	*/
  tfile = argv[5];
  pfile = argv[6];
  ofile = argv[7];

  read_input(temp, grid_rows, grid_cols, tfile);
  read_input(power, grid_rows, grid_cols, pfile);

  printf("Start computing the transient temperature\n");

  long long start_time = get_time();
  __mc_profiling_begin();
  compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols, chip_height, chip_width, t_chip, amb_temp, num_omp_threads);
  __mc_profiling_end();
  long long end_time = get_time();

  printf("Ending simulation\n");
  printf("Total time: %.3f seconds\n",
         ((float)(end_time - start_time)) / (1000 * 1000));

  writeoutput((1 & sim_time) ? result : temp, grid_rows, grid_cols, ofile);

  /* output results	*/
#ifdef VERBOSE
  fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
  for (i = 0; i < grid_rows * grid_cols; i++)
    fprintf(stdout, "%d\t%g\n", i, temp[i]);
#endif
  /* cleanup	*/
  free(temp);
  free(power);

  return 0;
}
/* vim: set ts=4 sw=4  sts=4 et si ai: */
