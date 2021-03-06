#ifndef GAUSSIAN_HEIGHT_FIELD_COLLISION_CUH
#define GAUSSIAN_HEIGHT_FIELD_COLLISION_CUH

#include <Vector3.h>
/*************************************************************************************************************************/
/*************************************************************************************************************************/
extern "C"
{
	void collisionSystem_Gaussian_HeightFieldCollision_CUDA(double* oldPos, 
				double* newPos, double* oldVel, double* newVel, double radiusParticle, float dt, uint nbBodiesP, 
				double A, double x0, double z0, double p1, double p2,
				Vector3 Min, Vector3 Max, float elast);
}
/*************************************************************************************************************************/
/*************************************************************************************************************************/
#endif
/*************************************************************************************************************************/
/*************************************************************************************************************************/
