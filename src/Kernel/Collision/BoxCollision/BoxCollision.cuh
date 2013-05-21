#include <Vector3.h>
/*****************************************************************************************************************/
/*****************************************************************************************************************/
extern "C"
{
void collisionSystem_Box_CUDA(double* newPos, double* oldVel, double* newVel, float radiusParticle, float dt, int nbBodiesP,
			      Vector3 Min, Vector3 Max, bool container, float elast);
}
/*****************************************************************************************************************/
/*****************************************************************************************************************/
