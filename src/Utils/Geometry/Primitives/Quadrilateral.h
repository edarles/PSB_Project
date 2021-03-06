#ifndef _QUADRILATERAL_
#define _QUADRILATERAL_

#include <Face.h>

using namespace std;

namespace Utils {

class Quadrilateral : public Face
{
	public:
		
		Quadrilateral();
		Quadrilateral(vector<Vector3> vertexs);
		Quadrilateral(Vector3 V0, Vector3 V1, Vector3 V2, Vector3 V3);
		Quadrilateral(const Quadrilateral&);
		~Quadrilateral();

		void create(Vector3 origin, float length, float width, Vector3 orientation);
};

}
#endif
