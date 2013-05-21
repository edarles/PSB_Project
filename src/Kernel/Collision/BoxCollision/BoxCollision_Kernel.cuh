#ifndef _COLLIDE_BOX_KERNEL_H_
#define _COLLIDE_BOX_KERNEL_H_

__global__ void collisionBox(int nbBodies, double3* newPos, double3* oldVel, double3* newVel, float3 Min, float3 Max,
			     float elast, bool container, float radiusParticle, float dt);

#endif
