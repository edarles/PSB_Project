#ifndef _COLLIDE_SPHERE_KERNEL_H_
#define _COLLIDE_SPHERE_KERNEL_H_

#include <common.cuh>

__host__ __device__ bool detection_collisionSphere(double3 oldPos, double3 newPos, float3 origin, float radius, bool container, double3 *pt_int, double3 *nInter, double *distance);

__global__ void collisionSphere(uint nbBodies, double3* newPos, double3* newVel, double3* oldPos, double3* oldVel, float3  origin, float radius, float elast, bool container, float dt);

#endif
