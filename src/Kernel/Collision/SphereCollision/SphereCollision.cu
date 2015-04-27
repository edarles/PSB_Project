#include <SphereCollision.cuh>
#include <SphereCollision_Kernel.cuh>
#include <common.cuh>

extern "C"
{
/**************************************************************************************************************************************/
void collisionSystem_Sphere_CUDA(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, float dt,
			         int nbBodies, Vector3 origin, float radius, float elast, bool container)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies, numBlocksX, numThreadsX);
  
    float3 O = make_float3(origin.x(),origin.y(),origin.z());
    collisionSphere<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*)newPos, (double3*)newVel,(double3*)oldPos, (double3*)oldVel, 
				           O, radius, elast, dt, container);
}

}
