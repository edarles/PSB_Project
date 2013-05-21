#ifndef _SPHERE_
#define _SPHERE_

#include <Vector3.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <ObjectGeo.h>
using namespace std;

namespace Utils {

class Sphere : public ObjectGeo
{
	public:
		
		Sphere();
		Sphere(Vector3 center, float radius);
		Sphere(const Sphere&);
		~Sphere();

		Vector3		getCenter();
		float		getRadius();

		void		setCenter(Vector3);
		void		setRadius(float);
		
		void 		display(Vector3 color);
		void 		displayNormale(Vector3 color);

	protected:

		Vector3 center;
		float   radius;
};

}
#endif
