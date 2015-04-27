#include <CylinderCollision.h>
#include <CylinderCollision.cuh>
/************************************************************************************/
/************************************************************************************/
CylinderCollision::CylinderCollision():ObjectCollision(),Cylinder()
{
}
/************************************************************************************/
CylinderCollision::CylinderCollision(float elast, bool is_container):ObjectCollision(elast,1-elast,is_container),Cylinder()
{
}
/************************************************************************************/
CylinderCollision::CylinderCollision(const CylinderCollision& C):ObjectCollision(C),Cylinder(C)
{
}
/************************************************************************************/
CylinderCollision::~CylinderCollision()
{
}
/************************************************************************************/
/************************************************************************************/
void CylinderCollision::create(Vector3 center, float baseRadius, float length, Vector3 direction, float elast, bool is_container)
{
	//direction.makeUnitVector();
	setElast(elast);
	setFriction(1-elast);
	setIsContainer(is_container);
	setCenter(center);
	setBaseRadius(baseRadius);
	setLength(length);
	setDirection(direction);
}
/************************************************************************************/
/************************************************************************************/
void CylinderCollision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		                float dt, int nbBodiesP)
{
	collisionCylinder_CUDA(newPos,newVel,radiusParticle,dt,nbBodiesP, 
			       getCenter(), getDirection(),getBaseRadius(),getLength(),getElast(),getIsContainer());
}
/************************************************************************************/
/************************************************************************************/
void CylinderCollision::display(GLenum mode, GLenum raster, Vector3 color)
{
	glPolygonMode(mode, raster);
	Cylinder::display(color);
}
/************************************************************************************/
void CylinderCollision::displayNormales(Vector3 color)
{
}
/************************************************************************************/
/************************************************************************************/
void CylinderCollision::inverseNormales()
{
}
/************************************************************************************/
/************************************************************************************/
