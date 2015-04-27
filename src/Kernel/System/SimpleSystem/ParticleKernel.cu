
#include <ParticleKernel.cuh>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>	 
#include <stdio.h>
#include <host_defines.h>
#include <double3.h>

// integrate particle attributes
__global__ void integrateEuler(uint nbBodies, double3* newPos, double3* newVel, 
   			       double3* forces, double* mass, float dt)
{
   uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x; 
   if(index < nbBodies){
	   double3 vel = newVel[index];
	   newVel[index] = newVel[index] + (forces[index]/mass[index]) * dt;
	   newPos[index] = newPos[index] + vel * dt;
	 
   }
}

