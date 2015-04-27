#ifndef PERIODIC_HEIGHT_FIELD_COLLISION_CUH
#define PERIODIC_HEIGHT_FIELD_COLLISION_CUH

#include <Vector3.h>
#include <common.cuh>
/*************************************************************************************************************************/
/*************************************************************************************************************************/
extern "C"
{
	void collisionSystem_Periodic_HeightFieldCollision_CUDA(double* oldPos, 
				double* newPos, double* oldVel, double* newVel, double radiusParticle, float dt, uint nbBodiesP, 
				uint nbFunc, double* A, double* k, double* theta, double* phi, 
				Vector3 Min, Vector3 Max, float elast);
}
/*************************************************************************************************************************/
/*************************************************************************************************************************/
#endif
/*************************************************************************************************************************/
/*************************************************************************************************************************/
