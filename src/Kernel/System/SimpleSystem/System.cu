#include <ParticleKernel.cuh>
#include <common.cuh>

extern "C"
{
void integrateSystem(double* oldPos, double* newPos, double* oldVel, double* newVel, 
		     double* forces, double* mass, double dt, int start, int numBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(numBodies,numBlocksX, numThreadsX);
    integrateEuler<<< numBlocksX, numThreadsX >>>(numBodies, (double3*)newPos, (double3*)newVel, (double3*) forces, mass, dt);
}

} 
