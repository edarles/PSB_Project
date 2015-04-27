#include <MSphParticle.h>
/********************************************************************************************************************/
/********************************************************************************************************************/
MSPHParticle::MSPHParticle():SPHParticle()
{
}
/********************************************************************************************************************/
MSPHParticle::MSPHParticle(Vector3 pos, Vector3 vel, double mass, float particleRadius, Vector3 color,
			   float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			   float gasStiffness, float threshold, float surfaceTension, float viscosity,
			   float temperature, Vector3 sigma, Vector3 beta, Vector3 g, Phase *p)

	     :SPHParticle(pos,vel,mass,particleRadius,color,interactionRadius,kernelParticles,density,restDensity,
			      pressure,gasStiffness,threshold,surfaceTension,viscosity)
{
	this->temperature = temperature;
	this->sigma = sigma;
	this->beta = beta;
	this->g = g;
	this->phase = p;
}
/********************************************************************************************************************/
MSPHParticle::MSPHParticle(Vector3 pos, Vector3 vel, Vector3 velInterAv, Vector3 velInterAp, double mass, float particleRadius, Vector3 color,
			   float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			   float gasStiffness, float threshold, float surfaceTension, float viscosity,
			   float temperature, Vector3 sigma, Vector3 beta, Vector3 g, Phase *p)

	      :SPHParticle(pos,vel,velInterAv,velInterAp, mass,particleRadius,color,interactionRadius,kernelParticles,density,restDensity,
			      pressure,gasStiffness,threshold,surfaceTension,viscosity)
{
	this->temperature = temperature;
	this->sigma = sigma;
	this->beta = beta;
	this->g = g;
	this->phase = p;
}
/********************************************************************************************************************/
MSPHParticle::MSPHParticle(const MSPHParticle& P):SPHParticle(P)
{
	this->temperature = P.temperature;
	this->sigma = P.sigma;
	this->beta = P.beta;
	this->g = P.g;
	this->phase = P.phase;
}
/********************************************************************************************************************/
MSPHParticle::~MSPHParticle()
{
}
/********************************************************************************************************************/
/********************************************************************************************************************/
Phase* MSPHParticle::getPhase()
{
	return phase;
}
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
void MSPHParticle::setPhase(Phase* p)
{
	this->phase = p;
}
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
