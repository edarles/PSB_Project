#ifndef _PARTICLE_LOADER_
#define _PARTICLE_LOADER_

#include <vector>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <Particle.h>
#include <System.h>
#include <SimpleSystem.h>
#include <CudaSystem.h>
#include <PciSphSystem.h>
#include <MSphSystem.h>

#include <CudaParticle.h>
#include <SphSystem.h>
#include <SphParticle.h>
#include <PciSphParticle.h>
#include <MSphParticle.h>

#include <typeinfo>

using namespace std;

namespace Utils {

class ParticleLoader
{
	public:
		ParticleLoader();
		~ParticleLoader();
	
		virtual vector<Particle*> load(System* S, const char *filename) = 0;
};

}
#endif 
