#include <MSphSystem.cuh>
#include <MSphSystem_Kernel.cuh>
#include <stdio.h>

extern "C"
{
/****************************************************************************************************************************/
void diffusionEvolution(double* dD, double* D0, double* dT, double EA, double* R, int nbBodies, int nbPhases, int indexPhase)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    diffusionEvolution_Kernel<<<numBlocksX,numThreadsX>>>(dD, D0, dT, EA, R, nbBodies, nbPhases, indexPhase);
    cudaDeviceSynchronize();
}

/****************************************************************************************************************************/
void temperatureEvolution(double* dT, double* D, double* mass, double* T, double* densBarre, double* pos, double* radius, partVoisine voisines, int* partPhases, 
			  int nbBodies, int nbPhases, int indexPhase)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    temperatureEvolution_Kernel<<<numBlocksX,numThreadsX>>>(dT, D, mass, T, densBarre, (double3*)pos, radius, voisines, partPhases, nbBodies, nbPhases, indexPhase);
    cudaDeviceSynchronize();
}

/****************************************************************************************************************************/
void viscosityEvolution(double* dMu, double* dT, double* T, double *dD, double* D, double* R, double* radius, double* densRest, int nbBodies, int nbPhases, int indexPhase)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    viscosityEvolution_Kernel<<<numBlocksX,numThreadsX>>>(dMu, dT, T, dD, D, R, radius, densRest, nbBodies, nbPhases, indexPhase);
    cudaDeviceSynchronize();
}
/****************************************************************************************************************************/
void concentrationEvolution(double* dAlphak, double* alphak, double *Dk, double* pos, double* mass, double* dens, double* radius, int* partPhases, partVoisine voisines, int nbBodies, int nbPhases, int indexPhase)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    evolutionConcentration_Kernel<<<numBlocksX,numThreadsX>>>(dAlphak, alphak, Dk, (double3*) pos, mass, dens, radius, partPhases, voisines, nbBodies, nbPhases, indexPhase);
    cudaDeviceSynchronize();
}
/****************************************************************************************************************************/
void viscosityEvolution2(double* dMu, double* dT, double* T, double *dD, double* D, double* R, double* radius, double* densRest, double* pos, double* mass, double* dens, partVoisine voisines,
					  int nbBodies, int nbPhases, int indexPhase)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    viscosityEvolution2_Kernel<<<numBlocksX,numThreadsX>>>(dMu, dT, T, dD, D, R, radius, densRest, (double3*)pos, mass, dens, voisines, nbBodies, nbPhases, indexPhase);
    cudaDeviceSynchronize();
}
/****************************************************************************************************************************/
void integrate_D_T_Mu(double* D, double* Dk, double *T, double* Tk, double *Mu, double *dD, double *dT, double *dMu, double *alpha, double TMin, double TMax, double muMin, double muMax,
		      double dt, int nbBodies, int nbPhases, int maxNumberPhases)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    integrate_D_T_Mu_Kernel<<<numBlocksX,numThreadsX>>>(D, Dk, T, Tk, Mu, dD, dT, dMu, alpha, TMin, TMax, muMin, muMax, dt, nbBodies, nbPhases, maxNumberPhases);
    cudaDeviceSynchronize();
}
/****************************************************************************************************************************/
void integrateConcentration(double* dAlphak, double* alphak, double dt, int nbBodies, int nbPhases, int maxNumberPhases)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    integrateConcentration_Kernel<<<numBlocksX,numThreadsX>>>(dAlphak, alphak, dt, nbBodies, nbPhases, maxNumberPhases);
    cudaDeviceSynchronize();
}
/****************************************************************************************************************************/
}
/****************************************************************************************************************************/

