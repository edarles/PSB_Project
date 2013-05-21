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
	if(typeid(*S) == typeid(SPHSystem)){
		SPHSystem *SPH = (SPHSystem*) S;
		if(SPH->getGridCreated()){
			SPH->_exportData_Mitsuba(filename);
		}
	}
}

}
