#ifndef __MSPH_SYSTEM_H__
#define __MSPH_SYSTEM_H__

#include <SphSystem.h>
#include <MSphParticle.h>

using namespace std;

#define maxPhases 4

class MSPHSystem : public SPHSystem
{
	public:

		static const double kB = 1.0;
		static const double EA = 2.0;
                static const double R = 3.0;

    		MSPHSystem();
    		virtual ~MSPHSystem();

		virtual void  addEmitter(Emitter* E);
   		virtual void  init();
    		virtual void  update();
    		virtual void  _initialize(int begin, int numParticles);
    		virtual void  _finalize();

		virtual void  displayParticlesByField(uint field);


		virtual void  _exportData_Mitsuba(const char* filenameDensity, const char* filenameAlbedo,
						  const char* filenameSigmaS, const char* filenameSigmaT);

	public:

		// Store temperatures and variations changes onto to GPU
		double  *m_d_T0k, *m_d_dTk, *m_d_dDk, *m_d_Dk, *m_d_Tk, *m_d_D0k, *m_d_dMuk, *m_d_Mu0k, *m_d_alphak, *m_d_dAlphak, *m_d_densRestk, *m_d_T, *m_d_D;
		double  *m_h_T0k, *m_h_dTk, *m_h_dDk, *m_h_Dk, *m_h_Tk, *m_h_D0k, *m_h_dMuk, *m_h_Mu0k, *m_h_alphak, *m_h_dAlphak, *m_h_densRestk, *m_h_T, *m_h_D;
		
		vector<Phase*> phases;
		int *m_d_partPhases, *m_h_partPhases;

		double tempMoy, MMoy, Rho0Moy;
		float tempMin, tempMax, massMin, massMax, muMin, muMax;
    
};

#endif //
