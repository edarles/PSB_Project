#include <MSphParticle.h>
/********************************************************************************************************************/
/********************************************************************************************************************/
MSPHParticle::MSPHParticle():PCI_SPHParticle()
{
}
/********************************************************************************************************************/
MSPHParticle::MSPHParticle(Vector3 pos, Vector3 vel, double mass, float particleRadius, Vector3 color,
			   float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			   float gasStiffness, float threshold, float surfaceTension, float viscosity,
			   float temperature,Vector3 sigma, Vector3 beta, Vector3 g)

	     :PCI_SPHParticle(pos,vel,mass,particleRadius,color,interactionRadius,kernelParticles,density,restDensity,
			      pressure,gasStiffness,threshold,surfaceTension,viscosity)
{
	this->temperature = temperature;
	this->sigma = sigma;
	this->beta = beta;
	this->g = g;
}
/********************************************************************************************************************/
MSPHParticle::MSPHParticle(const MSPHParticle& P):PCI_SPHParticle(P)
{
	this->temperature = P.temperature;
	this->sigma = P.sigma;
	this->beta = P.beta;
	this->g = P.g;
}
/********************************************************************************************************************/
~MSPHParticle::MSPHParticle()
{
}
/********************************************************************************************************************/
/********************************************************************************************************************/
float MSPHParticle::getTemperature()
{
	return temperature;
}
/********************************************************************************************************************/
Vector3 MSPHParticle::getSigma()
{
	return sigma;
}
/********************************************************************************************************************/
Vector3 MSPHParticle::getBeta()
{
	return beta;
}
/********************************************************************************************************************/
Vector3 MSPHParticle::getG()
{
	return g;
}
/********************************************************************************************************************/
/********************************************************************************************************************/
void MSPHParticle::setTemperature(float temperature)
{
	this->temperature = temperature;
}
/********************************************************************************************************************/
void MSPHParticle::setSigma(Vector3 sigma)
{
	this->sigma = sigma;
}
/********************************************************************************************************************/
void MSPHParticle::setBeta(Vector3 beta)
{
	this->beta = beta;
}
/********************************************************************************************************************/
void MSPHParticle::setG(Vector3 g)
{
	this->g = g;
}
/********************************************************************************************************************/
/********************************************************************************************************************/
