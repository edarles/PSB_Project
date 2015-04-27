#include <common.cuh>
#include <SimulationData_SPHSystem.h>
#include <SphKernel.cuh>

extern "C" {

/**********************************************************************************************************/
void integrateSPHSystem(double* velInterAv, double* velInterAp, double* oldPos, double* newPos, 
			double* forces, double* densities, double dt, int numBodies);

/**********************************************************************************************************/
void interpolateSPHVelocities(double* velInterAv, double* velInterAp, double* oldVel, double* newVel, int numBodies);

void postProcessCollide(double* velInterAv, double* velInterAp, double* oldVel, double* newVel, int nbBodies);
/**********************************************************************************************************/
void evaluate_densities_forces(double* pos, double* vel, double* mass, double* radius, double* densities, double* pressure, 
			       double* normales, double* restDensities, double* viscosities, double *k, double* threshold, double* surfaceTension, 
		               int numBodies, double* fPressure, double* fViscosity, double* fSurface, double* forcesAccum, partVoisine voisines);

/**********************************************************************************************************/
void evaluate_densitiesSPH(double* pos, double* mass, double* radius, double* densities, 
		        double* pressure, double* restDensities,
			double *k, int numBodies, partVoisine voisines);

/**********************************************************************************************************/
void evaluate_forcesSPH(double* pos, double* vel, double* mass, double* radius, double* densities, 
	             double* pressure, double* normales, double* viscosities, 
	             double* threshold, double* surfaceTension, 
		     int numBodies, double* fPressure, double* fViscosity, double* fSurface, 
	             double* forcesAccum, partVoisine voisines);
}
