#ifndef _TRIANGLE_
#define _TRIANGLE_

#include <Face.h>

using namespace std;

namespace Utils {

class Triangle : public Face
{
	public:
		
		Triangle();
		Triangle(vector<Vector3> vertexs);
		Triangle(const Triangle&);
		~Triangle();
};

}
#endif
