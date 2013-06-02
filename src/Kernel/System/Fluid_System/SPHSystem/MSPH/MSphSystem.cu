#include <MSphSystem.cuh>
#include <MSphSystem_Kernel.cuh>
#include <stdio.h>

extern "C"
{
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void evaluate_diffusion_coefficient(double* pos, double* mass, double* radius, 
				    double* density, double* ci, double* temperatures, double* visc, partVoisine voisines,
			            double* deltaCi, uint nbBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
  
    evaluate_diffusion_coefficient_Kernel<<<numBlocksX,numThreadsX>>>((double3*) pos, mass, radius, density, ci, temperatures, visc, 
								       voisines, deltaCi, nbBodies);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void evaluate_T_Rho0_Mass (double* pos, double* mass, double* radius, double* density,
			   double* restDensity, double* temp, double* cij, partVoisine voisines,
		           double* deltaT, double* deltaRho0, double* deltaM, uint nbBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
  
    evaluate_T_Rho0_Mass_Kernel<<<numBlocksX,numThreadsX>>>((double3*) pos, mass, radius, density, restDensity, temp, cij, voisines,
								      deltaT, deltaRho0, deltaM, nbBodies);

}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void integrate_T_Rho0_Mass(double *T, double* restDensity, double* mass,
			   double TS, double MS, double Rho0S,
		           double* dT, double* dRho0, double *dM, double dt, uint nbBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    integrate_T_Rho0_Mass_Kernel<<<numBlocksX,numThreadsX>>>(T, restDensity, mass, TS, MS, Rho0S, dT, dRho0, dM, dt, nbBodies);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
