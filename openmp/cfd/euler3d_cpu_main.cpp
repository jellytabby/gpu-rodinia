// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>

#ifdef OMP_OFFLOAD
#pragma omp declare target
#endif
#include <cmath>
#ifdef OMP_OFFLOAD
#pragma omp end declare target
#endif

#include <omp.h>
#include "euler3d_cpu_module.h"

extern "C" void __mc_profiling_begin(void);
extern "C" void __mc_profiling_end(void);
extern int block_length;

/*
 * Generic functions
 */
template <typename T>
T* alloc(int N)
{
	return new T[N];
}

template <typename T>
void dealloc(T* array)
{
	delete[] array;
}

#ifdef OMP_OFFLOAD
#pragma omp declare target
#endif
template <typename T>
void copy(T* dst, T* src, int N)
{
	#pragma omp parallel for default(shared) schedule(static)
	for(int i = 0; i < N; i++)
	{
		dst[i] = src[i];
	}
}
#ifdef OMP_OFFLOAD
#pragma omp end declare target
#endif


void dump(float* variables, int nel, int nelr)
{


	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << variables[i + VAR_DENSITY*nelr] << std::endl;
	}


	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++) file << variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
			file << std::endl;
		}
	}

	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
	}

}

void initialize_variables(int nelr, float* variables, float* ff_variable)
{
	#pragma omp parallel for default(shared) schedule(static)
	for(int i = 0; i < nelr; i++)
	{
		for(int j = 0; j < NVAR; j++) variables[i + j*nelr] = ff_variable[j];
	}
}

/*
 * Main function
 */
int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "specify data file name" << std::endl;
		return 0;
	}
	const char* data_file_name = argv[1];

        float ff_variable[NVAR];
        float3 ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy;

	// set far field conditions
	{
		const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

		ff_variable[VAR_DENSITY] = float(1.4);

		float ff_pressure = float(1.0f);
		float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / ff_variable[VAR_DENSITY]);
		float ff_speed = float(ff_mach)*ff_speed_of_sound;

		float3 ff_velocity;
		ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
		ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
		ff_velocity.z = 0.0f;

		ff_variable[VAR_MOMENTUM+0] = ff_variable[VAR_DENSITY] * ff_velocity.x;
		ff_variable[VAR_MOMENTUM+1] = ff_variable[VAR_DENSITY] * ff_velocity.y;
		ff_variable[VAR_MOMENTUM+2] = ff_variable[VAR_DENSITY] * ff_velocity.z;

		ff_variable[VAR_DENSITY_ENERGY] = ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

		float3 ff_momentum;
		ff_momentum.x = *(ff_variable+VAR_MOMENTUM+0);
		ff_momentum.y = *(ff_variable+VAR_MOMENTUM+1);
		ff_momentum.z = *(ff_variable+VAR_MOMENTUM+2);
		compute_flux_contribution(ff_variable[VAR_DENSITY], ff_momentum, ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy);
	}
	int nel;
	int nelr;


	// read in domain geometry
	float* areas;
	int* elements_surrounding_elements;
	float* normals;
	{
		std::ifstream file(data_file_name);

		file >> nel;
		nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));

		areas = new float[nelr];
		elements_surrounding_elements = new int[nelr*NNB];
		normals = new float[NDIM*NNB*nelr];

		// read in data
		for(int i = 0; i < nel; i++)
		{
			file >> areas[i];
			for(int j = 0; j < NNB; j++)
			{
				file >> elements_surrounding_elements[i + j*nelr];
				if(elements_surrounding_elements[i+j*nelr] < 0) elements_surrounding_elements[i+j*nelr] = -1;
				elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering

				for(int k = 0; k < NDIM; k++)
				{
					file >>  normals[i + (j + k*NNB)*nelr];
					normals[i + (j + k*NNB)*nelr] = -normals[i + (j + k*NNB)*nelr];
				}
			}
		}

		// fill in remaining data
		int last = nel-1;
		for(int i = nel; i < nelr; i++)
		{
			areas[i] = areas[last];
			for(int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				elements_surrounding_elements[i + j*nelr] = elements_surrounding_elements[last + j*nelr];
				for(int k = 0; k < NDIM; k++) normals[i + (j + k*NNB)*nelr] = normals[last + (j + k*NNB)*nelr];
			}
		}
	}

	// Create arrays and set initial conditions
	float* variables = alloc<float>(nelr*NVAR);
	initialize_variables(nelr, variables, ff_variable);

	float* old_variables = alloc<float>(nelr*NVAR);
	float* fluxes = alloc<float>(nelr*NVAR);
	float* step_factors = alloc<float>(nelr);

	// these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;
#ifdef _OPENMP
	double start = omp_get_wtime();
    #ifdef OMP_OFFLOAD
        #pragma omp target map(alloc: old_variables[0:(nelr*NVAR)]) map(to: nelr, areas[0:nelr], step_factors[0:nelr], elements_surrounding_elements[0:(nelr*NNB)], normals[0:(NDIM*NNB*nelr)], fluxes[0:(nelr*NVAR)], ff_variable[0:NVAR], ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy) map(variables[0:(nelr*NVAR)])
    #endif
#endif
	// Begin iterations
    __mc_profiling_begin();
	for(int i = 0; i < iterations; i++)
	{
                copy<float>(old_variables, variables, nelr*NVAR);

		// for the first iteration we compute the time step
		compute_step_factor(nelr, variables, areas, step_factors);

		for(int j = 0; j < RK; j++)
		{
			compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes, ff_variable, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy);
			time_step(j, nelr, old_variables, variables, step_factors, fluxes);
		}
	}
__mc_profiling_end();
#ifdef _OPENMP
	double end = omp_get_wtime();
	std::cout  << "Compute time: " << (end-start) << std::endl;
#endif


	std::cout << "Saving solution..." << std::endl;
	dump(variables, nel, nelr);
	std::cout << "Saved solution..." << std::endl;


	std::cout << "Cleaning up..." << std::endl;
	dealloc<float>(areas);
	dealloc<int>(elements_surrounding_elements);
	dealloc<float>(normals);

	dealloc<float>(variables);
	dealloc<float>(old_variables);
	dealloc<float>(fluxes);
	dealloc<float>(step_factors);

	std::cout << "Done..." << std::endl;

	return 0;
}
