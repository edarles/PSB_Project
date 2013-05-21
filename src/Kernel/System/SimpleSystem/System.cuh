#include <common.cuh>

extern "C" {

void integrateSystem(double* oldPos, double* newPos, double* oldVel, double* newVel, 
		     double* forces, double* mass, double dt, int start, int numBodies);

}
