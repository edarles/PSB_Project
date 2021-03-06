#ifndef _SPH_KERNEL_
#define _SPH_KERNEL_

extern "C" {

typedef struct
{
	int* nbVoisines;
	int* listeVoisine;
} partVoisine;

// DENSITY EVALUATION CUDA ROUTINE
__global__ void densityEvaluation(double3* pos, double* mass, double* radius, double *k, 
			          double* restDensity, uint nbBodies, double* density, double* pressure, partVoisine voisines);

// PRESSURE, VISCOSITY AND SURFACE TENSION FORCES EVALUATION CUDA ROUTINE
__global__ void internalForces(double3* pos, double3* vel, double* mass, double* density, double* pressure, 
			       double* radius, double* viscosity, double* l, double* surfaceTension, double3* normales, uint nbBodies,
			       double3* forcesPressure, double3* forcesViscosity, double3* forcesSurface, double3* forcesAccum, 
			       partVoisine voisines);

// INTEGRATION 
__global__ void integrateSPH_LeapFrog(double3* velInterAv, double3* velInterAp, double3* oldPos, double3* newPos, 
				      double3* forces, double*  densities, float dt, uint nbBodies);

__global__ void integrateSPH_LeapFrog_Forces(double3* velInterAv, double3* velInterAp, double3* oldPos, double3* newPos, 
				      double3* forcesViscosity, double3* forcesSurface, double3* forcesPressure, double3* forcesAccum,
				      double*  densities, float dt, uint nbBodies);

__global__ void interpolateSPH_velocities(double3* velInterAv, double3* velInterAp, double3* oldVel, double3* newVel, uint numBodies);
}
#endif
