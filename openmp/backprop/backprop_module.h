#define BIGRND 0x7fffffff

#define ETA 0.3      // eta value
#define MOMENTUM 0.3 // momentum value
#define NUM_THREAD 8 // OpenMP threads

typedef struct {
  int input_n;  /* number of input units */
  int hidden_n; /* number of hidden units */
  int output_n; /* number of output units */

  float *input_units;  /* the input units */
  float *hidden_units; /* the hidden units */
  float *output_units; /* the output units */

  float *hidden_delta; /* storage for hidden unit error */
  float *output_delta; /* storage for output unit error */

  float *target; /* storage for target vector */

  float **input_weights;  /* weights from input to hidden layer */
  float **hidden_weights; /* weights from hidden to output layer */

  /*** The next two are for momentum ***/
  float **input_prev_weights;  /* previous change on input to hidden wgt */
  float **hidden_prev_weights; /* previous change on hidden to output wgt */
} BPNN;

float squash(float x);

void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

void bpnn_output_error(float *delta, float *target, float *output, int nj,
                       float *err);

void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
                       float **who, float *hidden, float *err);
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
                         float **w, float **oldw);
