#include <SphSystem.cuh>
#include <cuda.h>
#include <Textures.cuh>
// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

extern "C"
{
/**********************************************************************************************************/
void integrateSPHSystem(double* velInterAv, double* velInterAp, double* oldPos, double* newPos, 
			double* forces, double* densities, double dt, int numBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(numBodies,numBlocksX, numThreadsX);
    integrateSPH_LeapFrog<<<numBlocksX, numThreadsX>>>((double3*) velInterAv, (double3*) velInterAp, (double3*) oldPos, (double3*) newPos, 
				                 (double3*) forces, densities, dt, numBodies);
    cudaDeviceSynchronize();
}

/**********************************************************************************************************/
void interpolateSPHVelocities(double* velInterAv, double* velInterAp, double* oldVel, double* newVel, int numBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(numBodies,numBlocksX, numThreadsX);
    interpolateSPH_velocities<<<numBlocksX, numThreadsX>>>
		((double3*) velInterAv, (double3*) velInterAp, (double3*) oldVel, (double3*) newVel, numBodies);
    cudaDeviceSynchronize();
}

/**********************************************************************************************************/
void postProcessCollide(double* velInterAv, double* velInterAp, double* oldVel, double* newVel, int numBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(numBodies,numBlocksX, numThreadsX);
    postProcessCollide_Kernel<<<numBlocksX, numThreadsX>>>
		((double3*) velInterAv, (double3*) velInterAp, (double3*) oldVel, (double3*) newVel, numBodies);
    cudaDeviceSynchronize();
}

/**********************************************************************************************************/
void evaluate_densitiesSPH(double* pos, double* mass, double* radius, double* densities, 
		        double* pressure, double* restDensities,  
			double *k, int numBodies, partVoisine voisines)
{
    int numThreadsX, numBlocksX;
    computeGridSize(numBodies,numBlocksX, numThreadsX);

    densityEvaluation<<<numBlocksX, numThreadsX>>>((double3*) pos, numBodies, radius, mass, k, restDensities, densities, pressure, voisines);
    cudaDeviceSynchronize();

}
/**********************************************************************************************************/
void evaluate_forcesSPH(double* pos, double* vel, double* mass, double* radius, double* densities, 
	             double* pressure, double* normales, double* viscosities, 
	             double* threshold, double* surfaceTension, 
		     int numBodies, double* fPressure, double* fViscosity, double* fSurface, 
	             double* forcesAccum, partVoisine voisines)
{
    int numThreadsX, numBlocksX;
    computeGridSize(numBodies,numBlocksX, numThreadsX);

    internalForces<<<numBlocksX, numThreadsX>>>((double3*) pos, (double3*) vel, mass, densities, pressure, 
			                  	radius, viscosities, threshold, surfaceTension,
				          	(double3*) normales, numBodies,
      			                  	(double3*) fPressure, (double3*) fViscosity, (double3*) fSurface, 
				          	(double3*) forcesAccum, voisines);
    cudaDeviceSynchronize();
}
/**********************************************************************************************************/
void evaluate_densities_forces(double* pos, double* vel, double* mass, double* radius, double* densities, 
			       double* pressure, double* normales, double* restDensities, double* viscosities, 
			       double *k, double* threshold, double* surfaceTension, 
		               int numBodies, double* fPressure, double* fViscosity, double* fSurface, 
			       double* forcesAccum, partVoisine voisines)
{
    int numThreadsX, numBlocksX;
    computeGridSize(numBodies,numBlocksX, numThreadsX);

    densityEvaluation<<<numBlocksX, numThreadsX>>>((double3*) pos, numBodies, radius, mass, k, restDensities, densities, pressure, voisines);
    cudaDeviceSynchronize();

    internalForces<<<numBlocksX, numThreadsX>>>((double3*) pos, (double3*) vel, mass, densities, pressure, 
			                  	radius, viscosities, threshold, surfaceTension,
				          	(double3*) normales, numBodies,
      			                  	(double3*) fPressure, (double3*) fViscosity, (double3*) fSurface, 
				          	(double3*) forcesAccum, voisines);
    cudaDeviceSynchronize();
}

}
