#ifndef _HEIGHT_FIELD__
#define _HEIGHT_FIELD__

#include <ObjectGeo.h>
#include <common.cuh>
using namespace std;

namespace Utils {

/*****************************************************************************************************/
/*****************************************************************************************************/
class HeightField : public ObjectGeo
{

/*****************************************************************************************************/
	public:
		HeightField();
		HeightField(Vector3 Min, Vector3 Max, double d_x, double d_z);
		HeightField(const HeightField& H);

		~HeightField();
/*****************************************************************************************************/
/*****************************************************************************************************/
		Vector3 getCenter();
		Vector3 getMin();
		Vector3 getMax();
		double  getDx();
		double  getDz();
/*****************************************************************************************************/
/*****************************************************************************************************/
		void create();
		void create(Vector3 min, Vector3 max, double dx, double dz);
/*****************************************************************************************************/
/*****************************************************************************************************/
		void generate();
/*****************************************************************************************************/
/*****************************************************************************************************/
		virtual void   calculateHeight(double* m_pos, uint nbPos) = 0;
		virtual void   calculateHeight_Normales(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1, uint nbPos) = 0;
/*****************************************************************************************************/
/*****************************************************************************************************/
		void 	display(Vector3 color);
		void 	displayNormale(Vector3 color);
/*****************************************************************************************************/
		void    exportToOBJ(const char* filename);
		void    exportToOBJ_noN(const char* filename);

	protected:
		Vector3 center, Min, Max;
		uint    nbPosX, nbPosY, nbPosZ;
		double  hMin, hMax;
		double    dx, dz;
		bool dis_norm;
		// Positions and normals CPU storage
		double *pos, *N;
		// Positions and normals GPU storage
		double *m_pos, *m_N;
/*****************************************************************************************************/
/*****************************************************************************************************/
};
/*****************************************************************************************************/
/*****************************************************************************************************/
}

#endif
