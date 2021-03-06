#ifndef _MESH_
#define _MESH_

#include <Vector3.h>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>
#include <Triangle.h>

#include <ObjectGeo.h>
using namespace std;

namespace Utils {

class Mesh : public ObjectGeo
{
	public:
		
		Mesh();
		Mesh(char* filename, vector<Triangle> faces);
		Mesh(const Mesh&);
		~Mesh();

		char*		   getFilename();
		vector<Triangle>   getFaces();
		Triangle           getFace(unsigned int);
		unsigned int       getNbFaces();

		void		   setFilename(char*);
		void 		   setFaces(vector<Triangle>);
		void		   setFace(unsigned int, Triangle Q);
		void		   addFace(Triangle T);

		void		   inverseNormales();

		void 		   display(Vector3 color);
		void 		   displayNormale(Vector3 color);

	protected:

		char* 		   filename;
		vector<Triangle>   faces;
};

}
#endif
