#ifndef _PARTICLE_LOADER_TXT_
#define _PARTICLE_LOADER_TXT_

#include <ParticleLoader.h>

using namespace std;

namespace Utils {

class ParticleLoader_Txt : public ParticleLoader
{
	public:
		ParticleLoader_Txt();
		~ParticleLoader_Txt();

		vector<Particle*> load(const char* filename);

};

}
#endif 
