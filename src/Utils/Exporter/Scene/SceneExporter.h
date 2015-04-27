#ifndef _SCENE_EXPORTER_
#define _SCENE_EXPORTER_


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

class SceneExporter 
{

  public:

	SceneExporter();
	~SceneExporter();

	virtual void _export(const char* filename, System *S) = 0;
};

}
#endif
