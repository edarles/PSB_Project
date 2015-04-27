#ifndef _MSPH_SYSTEM_CU_
#define _MSPH_SYSTEM_CU_

#include <MSphSystem_Kernel.cuh>
#include <cuda.h>

extern "C"
{
/****************************************************************************************************************************/
void diffusionEvolution(double* dD, double* D0, double* dT, double EA, double* R, int nbBodies, int nbPhases, int indexPhase);
/****************************************************************************************************************************/
void temperatureEvolution(double* dT, double* D, double* mass, double* T, double* densBarre, double* pos, double* radius, partVoisine voisines, int* partPhases, 
			  int nbBodies, int nbPhases, int indexPhase);
/****************************************************************************************************************************/
void viscosityEvolution(double* dMu, double* dT, double* T, double *dD, double* D, double *R, double* radius, double* densRest, int nbBodies, int nbPhases, int indexPhase);
void viscosityEvolution2(double* dMu, double* dT, double* T, double *dD, double* D, double* R, double* radius, double* densRest, double* pos, double* mass, double* dens, partVoisine voisines,
			 int nbBodies, int nbPhases, int indexPhase);
/****************************************************************************************************************************/
void concentrationEvolution(double* dAlphak, double* alphak, double *Dk, double* pos, double* mass, double* dens, double* radius, int* partPhases, partVoisine voisines, int nbBodies, int nbPhases, int indexPhase);
/****************************************************************************************************************************/
void integrate_D_T_Mu(double* D, double* Dk, double *T, double* Tk, double *Mu, double *dD, double *dT, double *dMu, double *alpha, double TMin, double TMax, double muMin, double muMax,
		      double dt, int nbBodies, int nbPhases, int maxNumberPhases);
/****************************************************************************************************************************/
void integrateConcentration(double* dAlphak, double* alphak, double dt, int nbBodies, int nbPhases, int maxNumberPhases);
}
#endif

