#ifndef _CUDA_PARTICLES_KERNEL_H_
#define _CUDA_PARTICLES_KERNEL_H_

#include <SphKernel.cuh>

extern "C"
{
__device__ double3 collideParticle(double3 posA, double3 posB,
                                  double3 velA, double3 velB,
                                  double radiusA, double radiusB,
                                  double spring, double damping, double shear, double attraction);

__global__ void collideParticles(uint nbBodies, double3* oldPos, double3* oldVel,
				 double3 *forces, double* radius, double* spring, 
				 double* damping, double* shear, double* attraction, partVoisine voisines);


}

#endif
