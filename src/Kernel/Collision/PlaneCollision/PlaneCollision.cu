#include <PlaneCollision.cuh>
#include <PlaneCollision_Kernel.cuh>
#include <common.cuh>
/**************************************************************************************************************************************/
/**************************************************************************************************************************************/
extern "C"
{
/**************************************************************************************************************************************/
void collisionPlan_CUDA(double* oldPos, double* newPos, double* oldVel, double* newVel, Vector3 A, Vector3 B, 
			Vector3 C, Vector3 D, Vector3 N, float elast, float dt, int nbBodiesP)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodiesP, numBlocksX, numThreadsX);
  
    float3 AF = make_float3(A.x(), A.y(), A.z());
    float3 BF = make_float3(B.x(), B.y(), B.z());
    float3 CF = make_float3(C.x(), C.y(), C.z());
    float3 DF = make_float3(D.x(), D.y(), D.z());
    float3 NF = make_float3(N.x(), N.y(), N.z());

    collisionPlan<<< numBlocksX, numThreadsX >>>(nbBodiesP, (double3*)newPos, (double3*)newVel,(double3*)oldPos, (double3*)oldVel, 
					         AF, BF, CF, DF, NF, elast, dt);
}
/**************************************************************************************************************************************/
}
