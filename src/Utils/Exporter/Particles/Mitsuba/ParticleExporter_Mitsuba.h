#ifndef _PARTICLE_EXPORTER_MITSUBA_
#define _PARTICLE_EXPORTER_MITSUBA_

#include <ParticleExporter.h>

namespace Utils {

class ParticleExporter_Mitsuba : public ParticleExporter {

 public:
	ParticleExporter_Mitsuba();
	~ParticleExporter_Mitsuba();

	void _export(const char* filename,  System *S);
	void _exportData(const char* filenameDensity, const char* filenameAlbedo, 
			 const char* filenameSigmaS, const char* filenameSigmaT, System* S);

 //private:
	//void _exportData(const char* filename, UniformGrid *grid);
};
}

#endif
