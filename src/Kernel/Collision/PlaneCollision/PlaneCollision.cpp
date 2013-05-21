/**********************************************************************************/
/**********************************************************************************/
#include <PlaneCollision.h>
#include <PlaneCollision.cuh>

/**********************************************************************************/
/**********************************************************************************/
PlaneCollision::PlaneCollision():ObjectCollision(),Quadrilateral()
{
}
/**********************************************************************************/
PlaneCollision::PlaneCollision(float elast, float friction, bool is_container):ObjectCollision(elast, friction, is_container),Quadrilateral()
{
}
/**********************************************************************************/
PlaneCollision::PlaneCollision(const PlaneCollision& O):ObjectCollision(O),Quadrilateral(O)
{
}
/**********************************************************************************/
PlaneCollision::~PlaneCollision()
{
}
/**********************************************************************************/
/**********************************************************************************/
void PlaneCollision::create(Vector3 origin, float length, float width, float elast, float friction, bool is_container, Vector3 orientation)
{
	setElast(elast);
	setFriction(friction);
	setIsContainer(is_container);
	Quadrilateral::create(origin,length,width,orientation);
       
}
/**********************************************************************************/
/**********************************************************************************/
void PlaneCollision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		              float dt, int nbBodiesP)
{
	collisionPlan_CUDA(oldPos,newPos,oldVel,newVel,getVertex(0),getVertex(1),getVertex(2),getVertex(3),getNormale(),getElast(),dt,nbBodiesP);
}
/**********************************************************************************/
/**********************************************************************************/
void PlaneCollision::display(GLenum mode, GLenum raster, Vector3 color)
{
	glPolygonMode(mode, raster);
	Quadrilateral::display(color);
}
/**********************************************************************************/
void PlaneCollision::displayNormales(Vector3 color)
{
	Quadrilateral::displayNormale(color);
}
/**********************************************************************************/
/**********************************************************************************/
