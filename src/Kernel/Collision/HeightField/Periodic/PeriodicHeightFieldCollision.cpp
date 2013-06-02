#include <PeriodicHeightFieldCollision.h>
#include <PeriodicHeightFieldCollision.cuh>
/***********************************************************************************/
/***********************************************************************************/
Periodic_HeightFieldCollision::Periodic_HeightFieldCollision()
			      :Periodic_HeightField(),HeightFieldCollision()
{
}
/***********************************************************************************/
Periodic_HeightFieldCollision::Periodic_HeightFieldCollision(float elast)
			      :Periodic_HeightField(),HeightFieldCollision(elast)
{
}
/***********************************************************************************/
Periodic_HeightFieldCollision::Periodic_HeightFieldCollision(const Periodic_HeightFieldCollision& H)
			      :Periodic_HeightField(H),HeightFieldCollision(H)
{
}
/***********************************************************************************/
Periodic_HeightFieldCollision::~Periodic_HeightFieldCollision()
{
}
/***********************************************************************************/
/***********************************************************************************/
void Periodic_HeightFieldCollision::create(Vector3 origin, float length, float width, double dx_, double dz_,
			    		   uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
			    		   double phiMin, double phiMax, float elast)
{
	setElast(elast);
	Vector3 min_(origin.x()-length/2,origin.y(),origin.z()-width/2);
	Vector3 max_(origin.x()+length/2,origin.y(),origin.z()+width/2);

	Periodic_HeightField::create(nbFunc,AMin,AMax,kMin,kMax,thetaMin,thetaMax,phiMin,phiMax,0,0);
	HeightField::create(min_,max_,dx_,dz_);
	HeightField::generate();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Periodic_HeightFieldCollision::collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             		    float dt, int nbBodiesP)
{
	collisionSystem_Periodic_HeightFieldCollision_CUDA(oldPos,newPos,oldVel,newVel,radiusParticle,dt,nbBodiesP,
							  nbFunc,m_A,m_k,m_theta,m_phi,Min,Max,elast);
}
/***********************************************************************************/
/***********************************************************************************/
void Periodic_HeightFieldCollision::loadSpectrum(const char* filename)
{
	Periodic_HeightField::loadSpectrum(filename);
	HeightField::create(Min,Max,dx,dz);
	HeightField::generate();
}
/***********************************************************************************/
/***********************************************************************************/
void Periodic_HeightFieldCollision::saveSpectrum(const char* filename)
{
	Periodic_HeightField::saveSpectrum(filename);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Periodic_HeightFieldCollision::display(GLenum mode, GLenum raster, Vector3 color)
{
	glPolygonMode(mode, raster);
	HeightField::display(color);
}
/*****************************************************************************************************/
void Periodic_HeightFieldCollision::displayNormales(Vector3 color)
{
	HeightField::displayNormale(color);
}
/*****************************************************************************************************/
/*****************************************************************************************************/

