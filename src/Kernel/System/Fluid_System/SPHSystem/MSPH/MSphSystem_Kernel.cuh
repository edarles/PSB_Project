#ifndef _MSPH_KERNEL_CUH
#define _MSPH_KERNEL_CUH

#include <common.cuh>
#include <SphKernel.cuh>

extern "C" {

/****************************************************************************************************************************/
__global__ void diffusionEvolution_Kernel(double* dD, double* D0, double* dT, double EA, double* R, int nbBodies, int nbPhases, int indexPhase);
/****************************************************************************************************************************/
__global__ void temperatureEvolution_Kernel(double* dT, double* D, double* mass, double* T, double* densBarre, double3* pos, double* radius, 
					    partVoisine voisines, int* partPhases, int nbBodies, int nbPhases, int indexPhase);
/****************************************************************************************************************************/
__global__ void viscosityEvolution_Kernel(double* dMu, double* dT, double* T, double *dD, double* D, double* R, double* radius, double* densRest, int nbBodies, int nbPhases, int indexPhase);
/****************************************************************************************************************************/
__global__ void evolutionConcentration_Kernel(double* dAlphak, double* alphak, double *Dk, double3* pos, double* mass, double* dens, double* radius, int* partPhases,
					      partVoisine voisines, int nbBodies, int nbPhases, int indexPhase);
/****************************************************************************************************************************/
__global__ void viscosityEvolution2_Kernel(double* dMu, double* dT, double* T, double *dD, double* D, double* R, double* radius, double* densRest, double3* pos, double* mass, 
					  double* dens, partVoisine voisines,
					  int nbBodies, int nbPhases, int indexPhase);
/****************************************************************************************************************************/
__global__ void integrate_D_T_Mu_Kernel(double* D, double* Dk, double *T, double* Tk, double *Mu, double *dD, double *dT, double *dMu, 
					double *alpha, double TMin, double TMax, double muMin, double muMax,
					double dt, int nbBodies, int nbPhases, int maxNumberPhases);
/****************************************************************************************************************************/
__global__ void integrateConcentration_Kernel(double* dAlphak, double* alphak, double dt, int nbBodies, int nbPhases, int maxNumberPhases);
}
#endif
