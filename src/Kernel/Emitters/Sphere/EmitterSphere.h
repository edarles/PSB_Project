#ifndef EMITTER_SPHERE_
#define EMITTER_SPHERE_

#include <Sphere.h>
#include <Emitter.h>

using namespace Utils;

class EmitterSphere : public Emitter, public Sphere
{
	public :
		EmitterSphere();
		EmitterSphere(const Sphere&, Vector3 velocityEmission);
		EmitterSphere(Vector3 center, float baseRadius, Vector3 velocityEmission);
		EmitterSphere(const EmitterSphere&);
		~EmitterSphere();

		vector<Particle*> emitParticles();
		vector<Particle*> emitParticles2D();
		vector<Particle*> emitParticles2D_z();
		void display(Vector3 color);

};

#endif

