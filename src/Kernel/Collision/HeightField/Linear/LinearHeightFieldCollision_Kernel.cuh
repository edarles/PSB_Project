#ifndef LINEAR_HEIGHT_FIELD_COLLISION_KERNEL
#define LINEAR_HEIGHT_FIELD_COLLISION_KERNEL

/*****************************************************************************************************/
/*****************************************************************************************************/
extern "C"
{
/*****************************************************************************************************/
	__host__ __device__ double  calculateHeight_Linear(double3 pos, double a, double b);

/*****************************************************************************************************/
	__host__ __device__ double3 approximateNormale_Linear(double3 pos, double a, double b);

/*****************************************************************************************************/
	__global__ void collisionSystem_Linear_HeightFieldCollision_Kernel
		   (double3* newPos, double3 *newVel, double radiusParticle, float dt, uint nbBodiesP, double a, double b, 
		    float3 min_, float3 max_, float elast);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
