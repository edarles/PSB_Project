#ifndef _PCI_SPH_SYSTEM_CU_
#define _PCI_SPH_SYSTEM_CU_

#include <PciSphSystem_Kernel.cuh>
#include <SphKernel.cuh>
#include <cuda.h>

extern "C"

{

void evaluate_densities_forcesVisc_PCI(double* pos, double* vel, double* mass, double* radius, double* densities, 
			       double* pressure, double* normales, double* restDensities, double* densityError, double* viscosities, 
			       double* threshold, double* surfaceTension, 
		               int numBodies, double* fViscosity, double* fSurface, double* fPressure, partVoisine voisines);

void integrate_PCI_SPHSystem(double* velAv, double *velAp, double* posAv, double* posAp, double* fV, double* fS, double* fP, double* fExt, double* densities, double dt, int nbBodies);


void pci_SPH_pressureForce(double* oldPos, double* pos, double* vel, double* velInterAv, double* velInterAp, 
		    double* mass, double* radius, double* density, double* restDensity, double* densityError, double* k, double* pressure,
		    double threshold, partVoisine voisines, double* forcesPressure, float dt, uint nbBodies);


}

#endif
