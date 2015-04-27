#ifndef _EMITTERS_
#define _EMITTERS_

#include <Emitter.h>
#include <EmitterBox.h>
#include <EmitterMesh.h>
#include <EmitterElipsoide.h>
#include <EmitterCylinder.h>
#include <EmitterSphere.h>
#include <EmitterGirly.h>
#include <vector>

using namespace std;

class Emitters {

	public:
		Emitters();
		~Emitters();

		void    	 reinit();

		vector<Emitter*> getEmitters();
		Emitter*	 getEmitter(unsigned int);
	
		void		 setEmitters(vector<Emitter*> emitters);
		void		 setEmitter(unsigned int, Emitter* E);
		void	         addEmitter(Emitter* E);

		vector<Particle*> emitParticles();
		vector<Particle*> emitParticles2D();
		vector<Particle*> emitParticles2D_z();

		void		  display(Vector3);

	private:
		vector<Emitter*> emitters;

};

#endif
