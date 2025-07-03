#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5
#define OPEN
typedef float FLOAT;

void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp,
                       FLOAT *power, int row, int col, FLOAT chip_height,
                       FLOAT chip_width, FLOAT t_chip, FLOAT amb_temp,
                       int omp_num_threads);

void single_iteration(FLOAT *result, FLOAT *temp, FLOAT *power, int row,
                      int col, FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1,
                      FLOAT step, FLOAT amb_temp, int num_omp_threads);
