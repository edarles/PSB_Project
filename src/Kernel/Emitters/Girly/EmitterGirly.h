#ifndef EMITTER_GIRLY_
#define EMITTER_GIRLY_

#include <Emitter.h>

// petit délire de geekette... :-)

class EmitterGirly : public Emitter
{
	public:
		EmitterGirly();
		EmitterGirly(Vector3 velocityEmission);
		EmitterGirly(const EmitterGirly&C);
		~EmitterGirly();

		vector<Particle*> emitParticles();
		void		  display(Vector3 color);
};

#endif

