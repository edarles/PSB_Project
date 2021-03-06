#ifndef EMITTER_MESH_
#define EMITTER_MESH_

#include <Mesh.h>
#include <Emitter.h>

using namespace Utils;

class EmitterMesh : public Emitter, public Mesh
{
	public :
		EmitterMesh();
		EmitterMesh(const Mesh&, unsigned int minEmission, unsigned int maxEmission, unsigned int durationTime, 
			    Vector3 velocityEmission);
		EmitterMesh(const EmitterMesh&);
		~EmitterMesh();

		vector<Particle*> emitParticles();
		void display(Vector3 color);

};

#endif

