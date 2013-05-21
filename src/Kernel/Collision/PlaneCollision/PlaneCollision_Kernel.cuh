#ifndef _COLLIDE_PLAN_KERNEL_H_
#define _COLLIDE_PLAN_KERNEL_H_

__host__ __device__ bool detectionPlan(float3 oldPos, float3 newPos, float3 A, float3 B, float3 C, float3 D, float3 N, float3 *pt_int, float3 *nInter, float *distance);

__global__ void collisionPlan(uint nbBodies, double3* newPos, double3* newVel, double3* oldPos, double3* oldVel, 
			      float3 AF, float3 BF, float3 CF, float3 DF, float3 NF, float elast, float dt);

#endif
