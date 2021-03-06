#ifndef COMBINED_HEIGHT_FIELD_COLLISION_KERNEL
#define COMBINED_HEIGHT_FIELD_COLLISION_KERNEL

/*****************************************************************************************************/
/*****************************************************************************************************/
extern "C"
{
/*****************************************************************************************************/
	__host__ __device__ double3 approximateNormale_Combined(double3 V1, double3 V2, double3 V3, double3 V4);

/*****************************************************************************************************/
	__global__ void collisionSystem_Combined_HeightFieldCollision_Kernel
		   (double3* newPos, double3 *newVel, double radiusParticle, float dt, uint nbBodiesP, 
		    double3* height, double3* normX0, double3* normX1, double3* normZ0, double3* normZ1,
		    float3 min_, float3 max_, float elast);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
