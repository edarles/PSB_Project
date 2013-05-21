#ifndef PARTICLE_
#define PARTICLE_

#include <Vector3.h>

class Particle {

	public:

		Particle();
		Particle(Vector3 pos, Vector3 vel, double mass, double particleRadius, Vector3 color);
		Particle(const Particle&);

		virtual ~Particle();

		Vector3 getOldPos();
		Vector3 getOldVel();
		Vector3 getNewPos();
		Vector3 getNewVel();
		double  getMass();
		double  getParticleRadius();
		Vector3 getColor();

		void setOldPos(Vector3);
		void setOldVel(Vector3);
		void setNewPos(Vector3);
		void setNewVel(Vector3);
		void setMass(double);
		void setParticleRadius(double);
		void setColor(Vector3);

	protected:

		Vector3 oldPos, newPos;
		Vector3 oldVel, newVel;
		double mass;
		double particleRadius;
		Vector3 color;
};

#endif
