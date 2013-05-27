#ifndef _PCI_SPH_SYSTEM_CU_
#define _PCI_SPH_SYSTEM_CU_

#include <PciSphSystem_Kernel.cuh>
#include <SphKernel.cuh>
#include <cuda.h>

extern "C"

{

void pci_SPH_update(double* oldPos, double* pos, double* vel, double* velInterAv, double* velInterAp, 
		    double* restore_oldPos, double* restore_pos, double* restore_velInterAv, double* restore_velInterAp,
		    double* mass, double* radius, double* density, double* restDensity, double* densityError, double* pressure,
		    double* k, double* viscosity, double* l, double* surfaceTension, double threshold, partVoisine voisines,
		    double* normales, double* forceViscosity, double* forceSurface, double* forcePressure, double* forceAccum,
		    float dt, uint nbBodies);

}

#endif
