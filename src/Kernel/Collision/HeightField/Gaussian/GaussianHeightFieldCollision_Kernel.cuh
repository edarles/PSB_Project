#ifndef GAUSSIAN_HEIGHT_FIELD_COLLISION_KERNEL
#define GAUSSIAN_HEIGHT_FIELD_COLLISION_KERNEL

#include <common.cuh>
/*****************************************************************************************************/
/*****************************************************************************************************/
extern "C"
{
/*****************************************************************************************************/
	__host__ __device__ double  calculateHeight_Gaussian(double3 pos, double A, double x0, double z0, double p1, double p2);

/*****************************************************************************************************/
	__host__ __device__ double3 approximateNormale_Gaussian(double3 pos, double A, double x0, double z0, double p1, double p2);

/*****************************************************************************************************/
	__global__ void collisionSystem_Gaussian_HeightFieldCollision_Kernel
		   (double3* newPos, double3 *newVel, double radiusParticle, float dt, uint nbBodiesP, 
		    double A, double x0, double z0, double p1, double p2,
		    float3 min_, float3 max_, float elast);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
