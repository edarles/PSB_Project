#include <LinearHeightFieldCollision.cuh>
#include <LinearHeightFieldCollision_Kernel.cuh>
#include <common.cuh>
#include <stdio.h>
/*******************************************************************************************************/
/*******************************************************************************************************/
extern "C"
{
void collisionSystem_Linear_HeightFieldCollision_CUDA(double* oldPos, 
			double* newPos, double* oldVel, double* newVel, double radiusParticle, float dt, uint nbBodiesP, double a, double b,
			Vector3 Min, Vector3 Max, float elast)
{
	int numThreadsX, numBlocksX;
        computeGridSize(nbBodiesP, numBlocksX, numThreadsX);
	
	float3 min_ = make_float3(Min.x(),Min.y(),Min.z());
	float3 max_ = make_float3(Max.x(),Max.y(),Max.z());

	collisionSystem_Linear_HeightFieldCollision_Kernel<<< numBlocksX,numThreadsX >>>
			((double3*)newPos, (double3*)newVel,radiusParticle,dt,nbBodiesP,a,b,min_,max_,elast);
}
/*******************************************************************************************************/
/*******************************************************************************************************/
}
/*******************************************************************************************************/
/*******************************************************************************************************/
