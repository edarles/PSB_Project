#include <Collision.h>
#include <stdio.h>
/************************************************************************************************/
/************************************************************************************************/
Collision::Collision()
{
}
/************************************************************************************************/
Collision::~Collision()
{
	for(uint i=0;i<objects.size();i++)
		delete(objects[i]);
	objects.clear();
}
/************************************************************************************************/
/************************************************************************************************/
ObjectCollision* Collision::getObject(unsigned int index)
{
	assert(index<objects.size());
	return objects[index];
}
/************************************************************************************************/
vector<ObjectCollision*> Collision::getObjects()
{
	return objects;
}
/************************************************************************************************/
/************************************************************************************************/
void Collision::removeLastObject()
{
	objects.pop_back();
}
/************************************************************************************************/
void Collision::setObject(unsigned int index, ObjectCollision *O)
{
	assert(index<objects.size());
	objects[index] = O;
}
/************************************************************************************************/
void Collision::setObject(ObjectCollision *O)
{
	objects.push_back(O);
}
/************************************************************************************************/
/************************************************************************************************/
void Collision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		         float dt, int nbBodiesP)
{
	  for(uint i=0;i<objects.size();i++)
		objects[i]->collide(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP);
	
}
/************************************************************************************************/
/************************************************************************************************/
void Collision::collide2D(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		         float dt, int nbBodiesP)
{
	for(uint i=0;i<objects.size();i++)
		objects[i]->collide2D(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP);
}
/************************************************************************************************/
/************************************************************************************************/
void Collision::display(GLenum mode, GLenum raster, Vector3 colorObject, Vector3 colorNormales, bool drawNormales)
{
	for(unsigned int i=0;i<objects.size();i++){
		objects[i]->display(mode, raster, colorObject);
		if(drawNormales)
			objects[i]->displayNormales(colorNormales);
	}
}
/************************************************************************************************/
/************************************************************************************************/
