#ifndef _PERIODIC_HEIGHT_FIELD__
#define _PERIODIC_HEIGHT_FIELD__

#include <HeightField.h>
using namespace std;

namespace Utils {

/*****************************************************************************************************/
/*****************************************************************************************************/
class Periodic_HeightField : public HeightField
{
/*****************************************************************************************************/
	public:
		Periodic_HeightField();
		Periodic_HeightField(uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
				     double phiMin, double phiMax, double wMin, double wMax, Vector3 Min, Vector3 Max, double d_x, double d_z);
		Periodic_HeightField(const Periodic_HeightField& H);
/*****************************************************************************************************/
		~Periodic_HeightField();
/*****************************************************************************************************/
/*****************************************************************************************************/
		void    create(uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
			       double phiMin, double phiMax, double wMin, double wMax);
/*****************************************************************************************************/
/*****************************************************************************************************/
		void calculateHeight(double* m_pos, uint nbPos);
		void calculateHeight_Normales_Gradient(double* m_pos, double* m_n, uint nbPos);
		void calculateHeight_Normales(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1, uint nbPos);
/*****************************************************************************************************/
/*****************************************************************************************************/
		void saveSpectrum(const char* filename);
		void loadSpectrum(const char* filename);
/*****************************************************************************************************/
/*****************************************************************************************************/
	protected:
		uint nbFunc;
		
		// GPU store
		double *m_A, *m_k, *m_theta, *m_phi, *m_w;

		double AMin, AMax, kMin, kMax, thetaMin, thetaMax, phiMin, phiMax, wMin, wMax;

		void initialize();
};
/*****************************************************************************************************/
/*****************************************************************************************************/
}
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
