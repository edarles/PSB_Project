#include <MeshCollision.cuh>
#include <MeshCollision_Kernel.cuh>
#include <common.cuh>
/***********************************************************************************************************/
/***********************************************************************************************************/
extern "C"
{
void collisionMesh_CUDA(double* oldPos, double* newPos, double* oldVel, double* newVel, 
			float radiusParticle, float dt, uint nbFaces, float* fA, float* fB, float* fC, float* fN, 
			float elast, int nbBodiesP)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodiesP, numBlocksX, numThreadsX);

    collisionTriangle<<<numBlocksX,numThreadsX>>>(nbBodiesP, nbFaces, (double3*)newPos, (double3*)newVel,
    					       (double3*)oldPos, (double3*)oldVel, 
					       fA, fB, fC, fN, elast, radiusParticle, dt);
    cudaDeviceSynchronize();
 }
/***********************************************************************************************************/
/***********************************************************************************************************/
}
/***********************************************************************************************************/
/***********************************************************************************************************/
