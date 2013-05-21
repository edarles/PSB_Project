#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

__global__ void integrateEuler(uint nbBodies, double3* newPos, double3* newVel, 
   			       double3* forces, double* mass, float dt);

#endif
