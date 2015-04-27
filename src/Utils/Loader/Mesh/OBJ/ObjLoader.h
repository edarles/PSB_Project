#ifndef _OBJ_LOADER_
#define _OBJ_LOADER_

#include <string>
#include <iostream>
#include <fstream>

#include <MeshLoader.h>

using namespace std;

namespace Utils {

class ObjLoader: public MeshLoader
{
	public:
		ObjLoader();
		~ObjLoader();

		Mesh load(const char *filename);
};

}
#endif 
