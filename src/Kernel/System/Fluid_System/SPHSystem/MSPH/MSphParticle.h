#ifndef _MSPH_PARTICLE_
#define _MSPH_PARTICLE_
/**************************************************************************************/
/**************************************************************************************/
#include <PciSphParticle.h>
#include <Phase.h>
/**************************************************************************************/
/**************************************************************************************/
class MSPHParticle : public SPHParticle {

	public:
/**************************************************************************************/
		MSPHParticle();
		
		MSPHParticle(Vector3 pos, Vector3 vel, double mass, float particleRadius, Vector3 color,
			     float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			     float gasStiffness, float threshold, float surfaceTension, float viscosity,
			     float temperature, Vector3 sigma, Vector3 beta, Vector3 g, Phase *p);

		MSPHParticle(Vector3 pos, Vector3 vel, Vector3 velInterAv, Vector3 velInterAp, double mass, float particleRadius, Vector3 color,
			     float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			     float gasStiffness, float threshold, float surfaceTension, float viscosity,
			     float temperature, Vector3 sigma, Vector3 beta, Vector3 g, Phase *p);

		MSPHParticle(const MSPHParticle&);
		~MSPHParticle();

/**************************************************************************************/
/**************************************************************************************/
		float      getTemperature();
		Phase*     getPhase();
		Vector3    getSigma();
		Vector3    getBeta();
		Vector3    getG();
/**************************************************************************************/
/**************************************************************************************/
		void  setTemperature(float temperature);
		void  setPhase(Phase *p);
		void  setSigma(Vector3 sigma);
		void  setBeta(Vector3 beta);
		void  setG(Vector3 g);

	private:
/**************************************************************************************/
		Phase *phase;
		float temperature;
/**************************************************************************************/
/*********** attribute for rendering **************************************************/
		Vector3 sigma, beta, g;
/**************************************************************************************/
/**************************************************************************************/
};
/**************************************************************************************/
/**************************************************************************************/
#endif
