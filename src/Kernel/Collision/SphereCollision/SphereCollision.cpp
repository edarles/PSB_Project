#include <SphereCollision.h>
#include <SphereCollision.cuh>

/***********************************************************************************/
/***********************************************************************************/
SphereCollision::SphereCollision():ObjectCollision(),Sphere()
{
}
/***********************************************************************************/
SphereCollision::SphereCollision(float elast, float friction, bool is_container):ObjectCollision(elast,friction, is_container),Sphere()
{
}
/***********************************************************************************/
SphereCollision::SphereCollision(const SphereCollision& Sc):ObjectCollision(Sc),Sphere(Sc)
{
}
/***********************************************************************************/
SphereCollision::SphereCollision(Vector3 origin, float radius, float elast, float friction, bool is_container):ObjectCollision(elast,friction,is_container),Sphere(origin,radius)
{
}
/***********************************************************************************/
SphereCollision::~SphereCollision()
{
}
/***********************************************************************************/
/***********************************************************************************/
void SphereCollision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		              float dt, int nbBodiesP)
{
	collisionSystem_Sphere_CUDA(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP,getCenter(),getRadius(),getElast(),getIsContainer());
}
/***********************************************************************************/
/***********************************************************************************/
void SphereCollision::display(GLenum mode, GLenum face, Vector3 color)
{
	glPolygonMode(mode, face);
	Sphere::display(color);
}
/***********************************************************************************/
void SphereCollision::displayNormales(Vector3 color)
{
	Sphere::displayNormale(color);
}
/***********************************************************************************/
