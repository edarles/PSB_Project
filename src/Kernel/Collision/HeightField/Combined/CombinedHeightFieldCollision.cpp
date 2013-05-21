#include <CombinedHeightFieldCollision.h>
#include <CombinedHeightFieldCollision.cuh>
#include <LinearHeightFieldCollision.h>
#include <PeriodicHeightFieldCollision.h>
#include <GaussianHeightFieldCollision.h>
#include <typeinfo>
#include <HeightField.cuh>
#include <common.cuh>
/*****************************************************************************************************/
/*****************************************************************************************************/
Combined_HeightFieldCollision::Combined_HeightFieldCollision():Combined_HeightField(),HeightFieldCollision()
{
}
/*****************************************************************************************************/
Combined_HeightFieldCollision::Combined_HeightFieldCollision(float elast):Combined_HeightField(),HeightFieldCollision(elast)
{
}
/*****************************************************************************************************/
Combined_HeightFieldCollision::Combined_HeightFieldCollision(const Combined_HeightFieldCollision& H)
			      :Combined_HeightField(H),HeightFieldCollision(H)
{
}
/*****************************************************************************************************/
Combined_HeightFieldCollision::~Combined_HeightFieldCollision()
{
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Combined_HeightFieldCollision::addHeightField(HeightField* H)
{
	Combined_HeightField::addHeightField(H);
}
/*****************************************************************************************************/
void Combined_HeightFieldCollision::setHeightField(HeightField* H, uint index)
{
	Combined_HeightField::setHeightField(H,index);
}
/*****************************************************************************************************/
void Combined_HeightFieldCollision::removeHeightFields(uint index)
{
	Combined_HeightField::removeHeightFields(index);
}
/*****************************************************************************************************/
void Combined_HeightFieldCollision::clearAll()
{
	Combined_HeightField::clearAll();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Combined_HeightFieldCollision::create(Vector3 origin, float length, float width, double dx_, double dz_, 
			                   vector<HeightField*> Hfields, float elast)
{
	setElast(elast);
	Vector3 min_(origin.x()-length/2,origin.y(),origin.z()-width/2);
	Vector3 max_(origin.x()+length/2,origin.y(),origin.z()+width/2);
	Combined_HeightField::create(Hfields);
	HeightField::create(min_,max_,dx_,dz_);
	generate();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Combined_HeightFieldCollision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             		    float dt, int nbBodiesP)
{
	double *pos, *normX0, *normX1, *normZ0, *normZ1;
	allocateArray((void**)&pos, nbBodiesP*sizeof(double)*3);
	allocateArray((void**)&normX0, nbBodiesP*sizeof(double)*3);
	allocateArray((void**)&normX1, nbBodiesP*sizeof(double)*3);
	allocateArray((void**)&normZ0, nbBodiesP*sizeof(double)*3);
	allocateArray((void**)&normZ1, nbBodiesP*sizeof(double)*3);

	copyArrayDeviceToDevice(pos, oldPos, 0, nbBodiesP*sizeof(double)*3);
	Combined_HeightField::calculateHeight_Normales(pos,normX0,normX1,normZ0,normZ1,nbBodiesP);		
	collisionSystem_Combined_HeightFieldCollision_CUDA(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP,
		pos,normX0,normX1,normZ0,normZ1,Min,Max,elast);

	freeArray(pos);
	freeArray(normX0);
	freeArray(normX1);
	freeArray(normZ0);
	freeArray(normZ1);	
/*
	for(uint i=0;i<HFields.size();i++){
			if(typeid(*HFields[i])==typeid(Linear_HeightFieldCollision)){
				Linear_HeightFieldCollision* H = (Linear_HeightFieldCollision*) HFields[i];
				H->collide(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP);
			}
			if(typeid(*HFields[i])==typeid(Gaussian_HeightFieldCollision)){
				Gaussian_HeightFieldCollision* H = (Gaussian_HeightFieldCollision*) HFields[i];
				H->collide(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP);
			}
			if(typeid(*HFields[i])==typeid(Periodic_HeightFieldCollision)){
				Periodic_HeightFieldCollision* H = (Periodic_HeightFieldCollision*) HFields[i];
				H->collide(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP);
			}
	}
*/				
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Combined_HeightFieldCollision::display(GLenum mode, GLenum raster, Vector3 color)
{
	glPolygonMode(mode, raster);
	HeightField::display(color);
}
/*****************************************************************************************************/
void Combined_HeightFieldCollision::displayNormales(Vector3 color)
{
	HeightField::displayNormale(color);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
