#ifndef _COLLIDE_FACE_KERNEL_H_
#define _COLLIDE_FACE_KERNEL_H_

extern "C"
{
__host__ __device__ bool pointInTriangle(float3 A, float3 B, float3 C, float3 P);

__host__ __device__ bool detectionTriangle(double3 oldPos, double3 newPos, float3 A, float3 B, float3 C, float3 N, 
					   double3 *pt_int, double3 *nInter, float *distance);

__global__ void collisionTriangle(uint nbBodies, uint nbFaces, double3* newPos, double3* newVel, double3* oldPos, double3* oldVel, 
			    	  float* AF, float* BF, float* CF, float* NF, float elast, float radiusParticle, float dt);

}
#endif
