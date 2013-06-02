#ifndef __MSPH_SYSTEM_H__
#define __MSPH_SYSTEM_H__

#include <SphSystem.h>
#include <MSphParticle.h>

using namespace std;

class MSPHSystem : public SPHSystem
{
	public:

    		MSPHSystem();
    		virtual ~MSPHSystem();

   		virtual void  init();
    		virtual void  update();
    		virtual void  _initialize(int begin, int numParticles);
    		virtual void  _finalize();

	public:
		// Store temperatures and variations changes onto to GPU
		double  *m_dTemp, *m_dCij, *m_dDeltaTemp, *m_dDeltaM, *m_dDeltaRho0;

		// Store temperatures onto to CPU
    		double *m_hTemp;

		double tempMoy, MMoy, Rho0Moy;
    
};

#endif //
