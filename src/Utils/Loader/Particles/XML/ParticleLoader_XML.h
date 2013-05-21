#ifndef _PARTICLE_LOADER_XML_
#define _PARTICLE_LOADER_XML_

#include <ParticleLoader.h>

using namespace std;

namespace Utils {

class ParticleLoader_XML : public ParticleLoader
{
	public:
		ParticleLoader_XML();
		~ParticleLoader_XML();

		vector<Particle*> load(const char *filename);

};

}
#endif 
