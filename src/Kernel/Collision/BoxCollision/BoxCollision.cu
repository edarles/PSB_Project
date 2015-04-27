#include <BoxCollision.cuh>
#include <BoxCollision_Kernel.cuh>
#include <common.cuh>
#include <stdio.h>
/*****************************************************************************************************************/
/*****************************************************************************************************************/
extern "C"
{
/**************************************************************************************************************************************/
void collisionSystem_Box_CUDA(double* newPos, double* oldPos, double* oldVel, double* newVel, float radiusParticle, float dt, int nbBodiesP,
			      Vector3 Min, Vector3 Max, bool container, float elast)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodiesP, numBlocksX, numThreadsX);
    collisionBox<<<numBlocksX,numThreadsX>>>(nbBodiesP, (double3*)newPos, (double3*)oldPos, (double3*) oldVel, (double3*)newVel, 
					make_float3(Min.x(),Min.y(),Min.z()), make_float3(Max.x(),Max.y(),Max.z()),
					elast, container, radiusParticle, dt);
    cudaDeviceSynchronize();
}
/**************************************************************************************************************************************/
void collisionSystem_Box_CUDA_2D(double* newPos, double* oldVel, double* newVel, float radiusParticle, float dt, int nbBodiesP,
			      Vector3 Min, Vector3 Max, bool container, float elast)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodiesP, numBlocksX, numThreadsX);
    collisionBox2D<<<numBlocksX,numThreadsX>>>(nbBodiesP, (double3*)newPos, (double3*) oldVel, (double3*)newVel, 
					make_float3(Min.x(),Min.y(),Min.z()), make_float3(Max.x(),Max.y(),Max.z()),
					elast, container, radiusParticle, dt);
    cudaDeviceSynchronize();
}
/*****************************************************************************************************************/
/*****************************************************************************************************************/
}
/*****************************************************************************************************************/
/*****************************************************************************************************************/
