#ifndef _PCISPH_PARTICLE_
#define _PCISPH_PARTICLE_

#include <SphParticle.h>

/**************************************************************************************/
/**************************************************************************************/
class PCI_SPHParticle : public SPHParticle {

	public:
/**************************************************************************************/
		PCI_SPHParticle();
		PCI_SPHParticle(Vector3 pos, Vector3 vel, double mass, float particleRadius, Vector3 color,
			 	float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			 	float gasStiffness, float threshold, float surfaceTension, float viscosity);

		PCI_SPHParticle(const PCI_SPHParticle&);

		~PCI_SPHParticle();
};
/**************************************************************************************/
/**************************************************************************************/
#endif
