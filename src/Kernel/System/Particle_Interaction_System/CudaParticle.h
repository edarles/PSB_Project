#ifndef _CUDA_PARTICLE_H
#define _CUDA_PARTICLE_H

#include <Particle.h>

class CudaParticle : public Particle
{
	public:
		CudaParticle();
		CudaParticle(Vector3 pos, Vector3 vel, double mass, double particleRadius, Vector3 color,
			   double interactionRadius, double spring, double damping, double shear, double attraction);
		CudaParticle(const CudaParticle&);
		~CudaParticle();
	
		double getInteractionRadius();
		double getSpring();
		double getDamping();
		double getShear();
		double getAttraction();

		void   setInteractionRadius(double);
		void   setSpring(double);
		void   setDamping(double);
		void   setShear(double);
		void   setAttraction(double);

	private:
		double interactionRadius;
		double spring;
		double damping;
		double shear;
		double attraction;
};

#endif
