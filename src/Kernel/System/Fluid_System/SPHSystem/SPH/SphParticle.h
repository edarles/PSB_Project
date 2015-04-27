#ifndef _SPH_PARTICLE_
#define _SPH_PARTICLE_

#include <Particle.h>

/**************************************************************************************/
/**************************************************************************************/
class SPHParticle : public Particle {

	public:
/**************************************************************************************/
		SPHParticle();
		SPHParticle(Vector3 pos, Vector3 vel, double mass, float particleRadius, Vector3 color,
			 float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			 float gasStiffness, float threshold, float surfaceTension, float viscosity);

		SPHParticle(Vector3 pos, Vector3 vel, Vector3 velInterAv, Vector3 velInterAp, 
			  double mass, float particleRadius, Vector3 color,
			 float interactionRadius, float kernelParticles, float density, float restDensity, float pressure,
			 float gasStiffness, float threshold, float surfaceTension, float viscosity);

		SPHParticle(const SPHParticle&);

		~SPHParticle();

/**************************************************************************************/
/**************************************************************************************/
		float	 getKernelParticles();
		float    getInteractionRadius();
		float    getDensity();
		float    getRestDensity();
		float    getPressure();
		float    getGasStiffness();
		float    getThreshold();
		float    getSurfaceTension();
		float    getViscosity();
		Vector3  getVelInterAv();
		Vector3  getVelInterAp();

/**************************************************************************************/
/**************************************************************************************/
		void  setKernelParticles(float kernelParticles);
		void  setInteractionRadius(float interactionRadius);
		void  setDensity(float density);
		void  setRestDensity(float restDensity);
		void  setPressure(float pressure);
		void  setGasStiffness(float gasStiffness);
		void  setThreshold(float threshold);
		void  setSurfaceTension(float surfaceTension);
		void  setViscosity(float viscosity);
		void  setVelInterAv(Vector3 velInterAv);
		void  setVelInterAp(Vector3 velInterAp);
	private:
/**************************************************************************************/
		float kernelParticles;
		float interactionRadius;
		float density;
		float restDensity;
		float pressure;
		float gasStiffness;
		float threshold;
		float surfaceTension;
		float viscosity;

		Vector3 velInterAv;
		Vector3 velInterAp;
/**************************************************************************************/
};
/**************************************************************************************/
/**************************************************************************************/
#endif
