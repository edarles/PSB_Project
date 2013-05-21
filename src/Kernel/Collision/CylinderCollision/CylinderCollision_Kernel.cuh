#ifndef _COLLIDE_CYLINDER_KERNEL_H_
#define _COLLIDE_CYLINDER_KERNEL_H_

#include <double3.h>

__global__ void collisionCylinder(int nbBodies, double3* newPos, double3* newVel, float radiusParticle, float dt,
			     	  float elast, bool container, float3 center, float baseRadius, float length, float3 direction);

#endif
