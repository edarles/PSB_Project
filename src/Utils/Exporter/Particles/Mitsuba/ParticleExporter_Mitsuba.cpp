#include <ParticleExporter_Mitsuba.h>
#include <typeinfo>
#include <stdio.h> 

namespace Utils {

ParticleExporter_Mitsuba::ParticleExporter_Mitsuba():ParticleExporter()
{
}
ParticleExporter_Mitsuba::~ParticleExporter_Mitsuba()
{
}
void ParticleExporter_Mitsuba::_export(const char* filename, System *S)
{
}

void ParticleExporter_Mitsuba::_exportData(const char* filenameDensity, const char* filenameAlbedo,
					   const char* filenameSigmaS, const char* filenameSigmaT, System *S)
{
	if(typeid(*S) == typeid(MSPHSystem)){
		MSPHSystem *SPH = (MSPHSystem*) S;
		if(SPH->getGridCreated()){
			SPH->_exportData_Mitsuba(filenameDensity,filenameAlbedo,filenameSigmaS,filenameSigmaT);
		}
	}
}

}
