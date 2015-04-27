#ifndef _EMITTER_ELIPSOIDE_
#define _EMITTER_ELIPSOIDE_

#include <Emitter.h>

class EmitterElipsoide : public Emitter 
{
	public:
		EmitterElipsoide();
		EmitterElipsoide(Vector3 center, float radius, double dx, double dy, double dz, 
				 unsigned int durationTime, Vector3 velocityEmission);

		EmitterElipsoide(const EmitterElipsoide&);
		~EmitterElipsoide();

		Vector3 getCenter();
		float   getRadius();
		Vector3 getDirection();

		void    setCenter(Vector3 center);
		void    setRadius(float radius);
		void    setDirection(Vector3 direction);

		vector<Particle*> emitParticles();
		vector<Particle*> emitParticles2D();
		vector<Particle*> emitParticles2D_z();

		void		  display(Vector3 color);

	private:
		Vector3 center, direction;
		float radius;
	
};
#endif
