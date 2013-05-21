#include <PciSphParticle.h>

/*********************************************************************************************************/
/*********************************************************************************************************/
PCI_SPHParticle::PCI_SPHParticle()
{
	temperature = 0;
}
/*********************************************************************************************************/
PCI_SPHParticle::PCI_SPHParticle(Vector3 pos, Vector3 vel, double mass, float particleRadius, Vector3 color,
			 	float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			 	float gasStiffness, float threshold, float surfaceTension, float viscosity, float temperature)
		:SPHParticle(pos,vel,mass,particleRadius,color,interactionRadius,kernelParticles,density,restDensity,pressure,gasStiffness,
			     threshold,surfaceTension,viscosity)
{
	this->temperature = temperature;
}
/*********************************************************************************************************/
PCI_SPHParticle::PCI_SPHParticle(const PCI_SPHParticle& P):SPHParticle(P)
{
	this->temperature = P.temperature;
}
/*********************************************************************************************************/
PCI_SPHParticle::~PCI_SPHParticle()
{
}
/*********************************************************************************************************/
/*********************************************************************************************************/
float PCI_SPHParticle::getTemperature()
{
	return temperature;
}
/*********************************************************************************************************/
void PCI_SPHParticle::setTemperature(float temperature)
{
	this->temperature = temperature;
}
/*********************************************************************************************************/
/*********************************************************************************************************/
