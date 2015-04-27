#ifndef _COLLIDE_FACE_KERNEL_H_
#define _COLLIDE_FACE_KERNEL_H_

#include <common.cuh>

extern "C"
{
__global__ void collisionTriangle(uint nbBodies, uint nbFaces, double3* newPos, double3* newVel, double3* oldPos, double3* oldVel, 
			    	  float* AF, float* BF, float* CF, float* NF, float elast, float radiusParticle, float dt);

}
#endif
