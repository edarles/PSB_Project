#include <PciSphParticle.h>

/*********************************************************************************************************/
/*********************************************************************************************************/
PCI_SPHParticle::PCI_SPHParticle():SPHParticle()
{
}
/*********************************************************************************************************/
PCI_SPHParticle::PCI_SPHParticle(Vector3 pos, Vector3 vel, double mass, float particleRadius, Vector3 color,
			 	float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			 	float gasStiffness, float threshold, float surfaceTension, float viscosity)
		:SPHParticle(pos,vel,mass,particleRadius,color,interactionRadius,kernelParticles,density,restDensity,pressure,gasStiffness,
			     threshold,surfaceTension,viscosity)
{
}
/*********************************************************************************************************/
PCI_SPHParticle::PCI_SPHParticle(const PCI_SPHParticle& P):SPHParticle(P)
{
}
/*********************************************************************************************************/
PCI_SPHParticle::~PCI_SPHParticle()
{
}
/*********************************************************************************************************/
/*********************************************************************************************************/
