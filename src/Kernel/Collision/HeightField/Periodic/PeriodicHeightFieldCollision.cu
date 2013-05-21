#include <PeriodicHeightFieldCollision.cuh>
#include <PeriodicHeightFieldCollision_Kernel.cuh>
#include <common.cuh>
/*******************************************************************************************************/
/*******************************************************************************************************/
extern "C"
{
void collisionSystem_Periodic_HeightFieldCollision_CUDA(double* oldPos, 
			double* newPos, double* oldVel, double* newVel, double radiusParticle, float dt, uint nbBodiesP, 
			uint nbFunc, double* A, double* k, double* theta, double* phi,
			Vector3 Min, Vector3 Max, float elast)
{
	int numThreadsX, numBlocksX;
        computeGridSize(nbBodiesP,numBlocksX, numThreadsX);
  
	float3 min_ = make_float3(Min.x(),Min.y(),Min.z());
	float3 max_ = make_float3(Max.x(),Max.y(),Max.z());

	collisionSystem_Periodic_HeightFieldCollision_Kernel<<< numBlocksX,numThreadsX >>>
			((double3*)newPos, (double3*)newVel,radiusParticle,dt,nbBodiesP,nbFunc,A,k,theta,phi,min_,max_,elast);
}
/*******************************************************************************************************/
/*******************************************************************************************************/
}
/*******************************************************************************************************/
/*******************************************************************************************************/
