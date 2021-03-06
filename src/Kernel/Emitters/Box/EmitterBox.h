#ifndef EMITTER_BOX_
#define EMITTER_BOX_

#include <Emitter.h>
#include <Box.h>
using namespace Utils;

class EmitterBox : public Emitter, public Box
{
	public:
		EmitterBox();
		EmitterBox(Vector3 center, float sizeX, float sizeY, float sizeZ, Vector3 velocityEmission);
		EmitterBox(const EmitterBox&C);
		~EmitterBox();

		vector<Particle*> emitParticles();
		void		  display(Vector3 color);

};

#endif

