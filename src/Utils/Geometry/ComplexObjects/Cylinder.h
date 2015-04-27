#ifndef _CYLINDER_
#define _CYLINDER_

#include <Vector3.h>
#include <ObjectGeo.h>

namespace Utils
{
class Cylinder : public ObjectGeo
{
	public:

		Cylinder();
		Cylinder(Vector3 center, float baseRadius,
			 float length,   Vector3 direction);
		Cylinder(const Cylinder&);

		~Cylinder();

		Vector3 getCenter();
		float	getBaseRadius();
		float   getLength();
		Vector3	getDirection();

		void	setCenter(Vector3);
		void	setBaseRadius(float);
		void	setLength(float);
		void	setDirection(Vector3);

		void	display(Vector3 color);
		void 	displayNormale(Vector3 color);

	protected:

		Vector3 center;
		float   baseRadius;
		float   length;
		Vector3 direction;
};

}
#endif
