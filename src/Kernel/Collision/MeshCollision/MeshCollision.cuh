#include <common.cuh>

extern "C"
{
 void collisionMesh_CUDA(double* oldPos, double* newPos, double* oldVel, double* newVel, 
			 float radiusParticle, float dt, uint nbFaces, float* fA, float* fB, float* fC, float* fN, 
			 float elast, int nbBodiesP);
}
