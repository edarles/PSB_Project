#ifndef _SURFACE_SPH_H__
#define _SURFACE_SPH_H__

#include <SphSystem.h>
#include <PciSphSystem.h>
#include <MSphSystem.h>

#include <Surface.h>
#include <vector>
using namespace std;

class SurfaceSPH : Surface {

	public:

		SurfaceSPH();
		SurfaceSPH(Vector3 color, double transparency);
		virtual ~SurfaceSPH();

		bool extract(System *S, double isoLevel, uint step);
		bool exportOBJ(const char* filename);
		void draw();
		void drawGrid();

	private:
		vector<Vector3> vertexs_cpu;
		vector<Vector3> normales_cpu;
		vector<int> indexs_cpu;
		double *grid;
		uint nbPosX,nbPosY,nbPosZ;
		float sizeCell;
};

#endif

