#ifndef PARTICLE_EXPORTER_XML_
#define PARTICLE_EXPORTER_XML_

#include <ParticleExporter.h>
#include <tinyxml.h>

namespace Utils {

class ParticleExporter_XML : public ParticleExporter {

 public:

	ParticleExporter_XML();
	~ParticleExporter_XML();

	void _export(const char* filename,  System *S);

 private:

	void _export_SimpleParticles(TiXmlDocument doc, TiXmlElement *elmPart, SimpleSystem *S);
	void _export_CudaParticles(TiXmlDocument doc, TiXmlElement *elmPart, CudaSystem *S);
	void _export_SPHParticles(TiXmlDocument doc, TiXmlElement *elmtPart, SPHSystem *S);
};

}

#endif
