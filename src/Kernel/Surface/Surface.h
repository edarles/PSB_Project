#ifndef _SURFACE_H__
#define _SURFACE_H__

#include <GL/gl.h>

#include <Vector3.h>
#include <System.h>

class Surface {

	public:

		Surface();
		Surface(Vector3 color, double transparency);
		virtual ~Surface();

		virtual bool extract(System *S, double isoLevel, uint step) = 0;
		virtual bool exportOBJ(const char* filename) = 0;
		virtual void draw() = 0;

	private:
		Vector3 color;
		double transparency;
};

#endif

