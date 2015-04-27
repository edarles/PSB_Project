#ifndef EMITTER_CYLINDER_
#define EMITTER_CYLINDER_

#include <Cylinder.h>
#include <Emitter.h>

using namespace Utils;

class EmitterCylinder : public Emitter, public Cylinder
{
	public :
		EmitterCylinder();
		EmitterCylinder(const Cylinder&, Vector3 velocityEmission);
		EmitterCylinder(Vector3 center, float baseRadius, float length, Vector3 direction, Vector3 velocityEmission);
		EmitterCylinder(const EmitterCylinder&);
		~EmitterCylinder();

		vector<Particle*> emitParticles();
		vector<Particle*> emitParticles2D();
		vector<Particle*> emitParticles2D_z();

		void display(Vector3 color);

};

#endif

