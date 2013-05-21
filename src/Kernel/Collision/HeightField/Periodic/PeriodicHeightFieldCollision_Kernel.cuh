#ifndef PERIODIC_HEIGHT_FIELD_COLLISION_KERNEL
#define PERIODIC_HEIGHT_FIELD_COLLISION_KERNEL

/*****************************************************************************************************/
/*****************************************************************************************************/
extern "C"
{
/*****************************************************************************************************/
	__host__ __device__ double  calculateHeight_Periodic(double3 pos, uint nbFunc, double* A, double* k, double* theta, double* phi);

/*****************************************************************************************************/
	__host__ __device__ double3 approximateNormale_Periodic(double3 pos, uint nbFunc, double* A, double* k, double* theta, double* phi);

/*****************************************************************************************************/
	__global__ void collisionSystem_Periodic_HeightFieldCollision_Kernel
		   (double3* newPos, double3 *newVel, double radiusParticle, float dt, uint nbBodiesP, 
		    uint nbFunc, double* A, double* k, double* theta, double* phi,
		    float3 min_, float3 max_, float elast);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
