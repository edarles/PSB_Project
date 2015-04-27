#ifndef SCENE_EXPORTER_XML_
#define SCENE_EXPORTER_XML_

#include <SceneExporter.h>
#include <tinyxml.h>

namespace Utils {

class SceneExporter_XML : public SceneExporter {

 public:
	SceneExporter_XML();
	~SceneExporter_XML();

	void _export(const char* filename,  System *S);

};

}

#endif
