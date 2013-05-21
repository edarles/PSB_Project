#ifndef _SCENE_EXPORTER_
#define _SCENE_EXPORTER_

#include <SimpleSystem.h>
#include <CudaSystem.h>

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
