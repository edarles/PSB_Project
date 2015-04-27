#ifndef _MESH_LOADER_
#define _MESH_LOADER_

#include <vector>
#include <Vector3.h>
#include <Vector2.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <Mesh.h>

using namespace std;

namespace Utils {

class MeshLoader
{
	public:
		MeshLoader();
		~MeshLoader();

		virtual Mesh load(const char *filename) = 0;
};

}
#endif 
