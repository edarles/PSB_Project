#ifndef _PARTICLE_EXPORTER_
#define _PARTICLE_EXPORTER_

#include <SimpleSystem.h>
#include <CudaSystem.h>
#include <SphSystem.h>
#include <PciSphSystem.h>
#include <MSphSystem.h>

#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>
#include <MSphParticle.h>

namespace Utils {

class ParticleExporter 
{

  public:

	ParticleExporter();
	~ParticleExporter();

	virtual void _export(const char* filename, System *S) = 0;
};

}
#endif
