#include <CudaParticle.h>

/*****************************************************************************************************************************************/
/*****************************************************************************************************************************************/
CudaParticle::CudaParticle():Particle()
{
	interactionRadius = 0 ;
	spring = 0;
	damping = 0;
	shear = 0;
	attraction = 0;
}
/*****************************************************************************************************************************************/
CudaParticle::CudaParticle(Vector3 pos, Vector3 vel, double mass, double particleRadius, Vector3 color,
			   double interactionRadius, double spring, double damping, double shear, double attraction)
	     :Particle(pos, vel, mass, particleRadius, color)
{
	this->interactionRadius = interactionRadius;
	this->spring = spring;
	this->damping = damping;
	this->shear = shear;
	this->attraction = attraction;
}
/*****************************************************************************************************************************************/
CudaParticle::CudaParticle(const CudaParticle& C):Particle(C)
{
	this->interactionRadius = C.interactionRadius;
	this->spring = C.spring;
	this->damping = C.damping;
	this->shear = C.shear;
	this->attraction = C.attraction;
}
/*****************************************************************************************************************************************/
CudaParticle::~CudaParticle()
{
}
/*****************************************************************************************************************************************/
/*****************************************************************************************************************************************/
double CudaParticle::getInteractionRadius()
{
	return interactionRadius;
}
/*****************************************************************************************************************************************/
double CudaParticle::getSpring()
{
	return spring;
}
/*****************************************************************************************************************************************/
double CudaParticle::getDamping()
{
	return damping;
}
/*****************************************************************************************************************************************/
double CudaParticle::getShear()
{
	return shear;
}
/*****************************************************************************************************************************************/
double CudaParticle::getAttraction()
{
	return attraction;
}
/*****************************************************************************************************************************************/
/*****************************************************************************************************************************************/
void   CudaParticle::setInteractionRadius(double interactionRadius)
{
	this->interactionRadius = interactionRadius;
}
/*****************************************************************************************************************************************/
void   CudaParticle::setSpring(double spring)
{
	this->spring = spring;
}
/*****************************************************************************************************************************************/
void   CudaParticle::setDamping(double damping)
{
	this->damping = damping;
}
/*****************************************************************************************************************************************/
void   CudaParticle::setShear(double shear)
{
	this->shear = shear;
}
/*****************************************************************************************************************************************/
void  CudaParticle::setAttraction(double attraction)
{
	this->attraction = attraction;
}
/*****************************************************************************************************************************************/
/*****************************************************************************************************************************************/