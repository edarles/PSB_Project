#include <CudaParticle_Kernel.cuh>
#include <common.cuh>

extern "C"
{
void interactionSystem(double* oldPos, double* oldVel, double* forces, 
		       double* interactionRadius, double* spring, double* damping,
		       double* shear, double* attraction, partVoisine voisines, int numBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(numBodies, numBlocksX, numThreadsX);
    collideParticles<<<numBlocksX, numThreadsX>>>(numBodies, (double3*)oldPos, (double3*)oldVel, (double3*) forces,
					    interactionRadius,spring,damping, shear,attraction, voisines );
}

}
