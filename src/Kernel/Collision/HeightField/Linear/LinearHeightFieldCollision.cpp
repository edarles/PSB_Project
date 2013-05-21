#include <LinearHeightFieldCollision.h>
#include <LinearHeightFieldCollision.cuh>
/*****************************************************************************************************/
/*****************************************************************************************************/
Linear_HeightFieldCollision::Linear_HeightFieldCollision():Linear_HeightField(),HeightFieldCollision()
{
}
/*****************************************************************************************************/
Linear_HeightFieldCollision::Linear_HeightFieldCollision(float elast):Linear_HeightField(),HeightFieldCollision(elast)
{
}
/*****************************************************************************************************/
Linear_HeightFieldCollision::Linear_HeightFieldCollision(const Linear_HeightFieldCollision& H):Linear_HeightField(H),HeightFieldCollision(H)
{
}
/*****************************************************************************************************/
Linear_HeightFieldCollision::~Linear_HeightFieldCollision()
{
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Linear_HeightFieldCollision::create(Vector3 origin, float length, float width, double dx_, double dz_,
					 double a, double b, float elast)
{
	setElast(elast);
	Vector3 min_(origin.x()-length/2,origin.y(),origin.z()-width/2);
	Vector3 max_(origin.x()+length/2,origin.y(),origin.z()+width/2);
	Linear_HeightField::create(a,b);
	HeightField::create(min_,max_,dx_,dz_);
	HeightField::generate();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Linear_HeightFieldCollision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             		  float dt, int nbBodiesP)
{
	collisionSystem_Linear_HeightFieldCollision_CUDA(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP,a,b,
			Min,Max,elast);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Linear_HeightFieldCollision::display(GLenum mode, GLenum raster, Vector3 color)
{
	glPolygonMode(mode, raster);
	HeightField::display(color);
}
/*****************************************************************************************************/
void Linear_HeightFieldCollision::displayNormales(Vector3 color)
{
	HeightField::displayNormale(color);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
