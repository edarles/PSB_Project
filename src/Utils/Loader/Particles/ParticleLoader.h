#ifndef _PARTICLE_LOADER_
#define _PARTICLE_LOADER_

#include <vector>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <Particle.h>

using namespace std;

namespace Utils {

class ParticleLoader
{
	public:
		ParticleLoader();
		~ParticleLoader();
	
		virtual vector<Particle*> load(const char *filename) = 0;
};

}
#endif 
