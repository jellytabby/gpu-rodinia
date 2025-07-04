// #define block_length omp_get_max_threads()

/*
 * Options
 *
 */
#define GAMMA 1.4
#define iterations 2000

#define NDIM 3
#define NNB 4

#define RK 3 // 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)

#ifdef restrict
#define __restrict restrict
#else
#define __restrict
#endif

struct float3 {
  float x, y, z;
};

void compute_flux_contribution(float &density, float3 &momentum,
                               float &density_energy, float &pressure,
                               float3 &velocity, float3 &fc_momentum_x,
                               float3 &fc_momentum_y, float3 &fc_momentum_z,
                               float3 &fc_density_energy);

void compute_flux(int nelr, int *elements_surrounding_elements, float *normals,
                  float *variables, float *fluxes, float *ff_variable,
                  float3 ff_flux_contribution_momentum_x,
                  float3 ff_flux_contribution_momentum_y,
                  float3 ff_flux_contribution_momentum_z,
                  float3 ff_flux_contribution_density_energy);

void time_step(int j, int nelr, float *old_variables, float *variables,
               float *step_factors, float *fluxes);

void compute_step_factor(int nelr, float *__restrict variables, float *areas,
                         float *__restrict step_factors);
