#include <CylinderCollision.cuh>
#include <CylinderCollision_Kernel.cuh>
#include <common.cuh>
/*****************************************************************************************************************/
/*****************************************************************************************************************/
extern "C"
{
/*****************************************************************************************************************/
/*****************************************************************************************************************/
void collisionCylinder_CUDA(double* newPos,  double* newVel, float radiusParticle, float dt, int nbBodiesP, 
			    Vector3 center, Vector3 direction, double radius, double length, float elast, bool container)
{

    int numThreadsX, numBlocksX;
    computeGridSize(nbBodiesP, numBlocksX, numThreadsX);
  
    float3 dir = make_float3(direction.x(),direction.y(),direction.z());
    float3 cent = make_float3(center.x(),center.y(),center.z());

    collisionCylinder<<<numBlocksX,numThreadsX >>>(nbBodiesP, (double3*)newPos, (double3*)newVel, radiusParticle, dt,
			     	  elast,container, cent, radius, length, dir);
    //CUT_CHECK_ERROR("collide kernel execution failed");
}
/*****************************************************************************************************************/
/*****************************************************************************************************************/
}
/*****************************************************************************************************************/
/*****************************************************************************************************************/
