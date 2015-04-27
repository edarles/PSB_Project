#include <BoxCollision.h>
#include <BoxCollision.cuh>
/**********************************************************************************************/
/**********************************************************************************************/
BoxCollision::BoxCollision():ObjectCollision(),Box()
{
}
/**********************************************************************************************/
BoxCollision::BoxCollision(float elast, bool is_container):ObjectCollision(elast, 1-elast, is_container),Box()
{
}
/**********************************************************************************************/
BoxCollision::BoxCollision(const BoxCollision& O):ObjectCollision(O),Box(O)
{
}
/**********************************************************************************************/
BoxCollision::~BoxCollision()
{
}
/**********************************************************************************************/
/**********************************************************************************************/
void BoxCollision::create(Vector3 origin, float length, float width, float depth, float restitution, bool is_container)
{
	setElast(restitution);
	setFriction(1-restitution);
	setIsContainer(is_container);
	Box::create(origin,length,depth,width);
	if(is_container)
		inverseNormales();
}
/**********************************************************************************************/
/**********************************************************************************************/
void BoxCollision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel,float radiusParticle, 
			   float dt, int nbBodiesP)
{
    collisionSystem_Box_CUDA(newPos,oldPos,oldVel,newVel,radiusParticle,dt,nbBodiesP,getMin(),getMax(),getIsContainer(),getElast()); 
}
/**********************************************************************************************/
/**********************************************************************************************/
void BoxCollision::collide2D(double* oldPos, double* newPos, double* oldVel, double* newVel,float radiusParticle, 
			   float dt, int nbBodiesP)
{
    collisionSystem_Box_CUDA_2D(newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP,getMin(),getMax(),getIsContainer(),getElast()); 
}
/**********************************************************************************************/
/**********************************************************************************************/
void BoxCollision::inverseNormales()
{
	for(unsigned int i=0;i<faces.size();i++)
			faces[i].setNormale(faces[i].getNormale()*Vector3(-1,-1,-1));
}
/**********************************************************************************************/
/**********************************************************************************************/
void BoxCollision::display(GLenum mode, GLenum raster, Vector3 color)
{
	glPolygonMode(mode, raster);
	Box::display(color);
	//Box::display(Vector3(0,0,0));
}
/**********************************************************************************************/
void BoxCollision::displayNormales(Vector3 color)
{
	Box::displayNormale(color);
}
/**********************************************************************************************/
/**********************************************************************************************/
