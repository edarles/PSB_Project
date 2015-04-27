#ifndef _FACE_
#define _FACE_

#include <Vector3.h>
#include <vector>
#include <GL/gl.h>
#include <GL/glu.h>

#include <ObjectGeo.h>

using namespace std;

namespace Utils {

class Face : public ObjectGeo
{
	public:
		
		Face();
		Face(vector<Vector3> vertexs);
		Face(const Face&);
		~Face();

		vector<Vector3> getVertexs();
		Vector3         getVertex(unsigned int);
		Vector3		getNormale();

		void	        setVertex(unsigned int, Vector3);
		void  		setVertexs(vector<Vector3> vertexs);
		void	        setVertex(Vector3);	

		void 		setNormale(Vector3);

		virtual void 	display(Vector3 color);
		virtual void 	displayNormale(Vector3 color);

		virtual void	calculateNormale();
			void 	inverseNormale();

	protected:

		vector<Vector3> vertexs;
		Vector3 N;
};

}
#endif
