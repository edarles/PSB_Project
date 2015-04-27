#include <GaussianHeightFieldCollision.h>
#include <GaussianHeightFieldCollision.cuh>
/*****************************************************************************************************/
/*****************************************************************************************************/
Gaussian_HeightFieldCollision::Gaussian_HeightFieldCollision():Gaussian_HeightField(),HeightFieldCollision()
{
}
/*****************************************************************************************************/
Gaussian_HeightFieldCollision::Gaussian_HeightFieldCollision(float elast):Gaussian_HeightField(),HeightFieldCollision(elast)
{
}
/*****************************************************************************************************/
Gaussian_HeightFieldCollision::Gaussian_HeightFieldCollision(const Gaussian_HeightFieldCollision& H)
			      :Gaussian_HeightField(H),HeightFieldCollision(H)
{
}
/*****************************************************************************************************/
Gaussian_HeightFieldCollision::~Gaussian_HeightFieldCollision()
{
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Gaussian_HeightFieldCollision::create(Vector3 origin, float length, float width, double dx_, double dz_,
					   double A, double p1, double p2, float elast)
{
	setElast(elast);
	Vector3 min_(origin.x()-length/2,origin.y(),origin.z()-width/2);
	Vector3 max_(origin.x()+length/2,origin.y(),origin.z()+width/2);

	Gaussian_HeightField::create(A,p1,p2);
	HeightField::create(min_,max_,dx_,dz_);
	HeightField::generate();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Gaussian_HeightFieldCollision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             		    float dt, int nbBodiesP)
{
	collisionSystem_Gaussian_HeightFieldCollision_CUDA(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP,A,
			center.x(),
			center.z(),p1,p2,Min,
			Max,elast);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Gaussian_HeightFieldCollision::display(GLenum mode, GLenum raster, Vector3 color)
{
	glPolygonMode(mode, raster);
	HeightField::display(color);
}
/*****************************************************************************************************/
void Gaussian_HeightFieldCollision::displayNormales(Vector3 color)
{
	HeightField::displayNormale(color);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
