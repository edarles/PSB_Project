#include <common.cuh>
#include <SimulationData_CudaSystem.h>

extern "C" {

void interactionSystem(double* oldPos, double* oldVel, double* forces, 
		       double* interactionRadius, double* spring, double* damping,
		       double* shear, double* attraction, partVoisine voisines, int numBodies);

}
