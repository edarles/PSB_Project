#ifndef PARTICLE_EXPORTER_TXT_
#define PARTICLE_EXPORTER_TXT_

#include <ParticleExporter.h>

namespace Utils {

class ParticleExporter_Txt : public ParticleExporter {

 public:
	ParticleExporter_Txt();
	~ParticleExporter_Txt();

	void _export(const char* filename,  System *S);

};

}

#endif
