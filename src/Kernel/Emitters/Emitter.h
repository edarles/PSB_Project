#ifndef _EMITTER_
#define _EMITTER_

#include <Particle.h>
#include <vector>
#include <SimulationData.h>
#include <SimulationData_SimpleSystem.h>
#include <SimulationData_CudaSystem.h>
#include <SimulationData_SPHSystem.h>
#include <SimulationData_PCI_SPHSystem.h>
#include <Vector3.h>
#include <GL/gl.h>
#include <GL/glu.h>

using namespace std;

class Emitter {

	public:
		Emitter();
		Emitter(Vector3 worldPosition, unsigned int minEmission, unsigned int maxEmission, 
			unsigned int durationTime, Vector3 velocityEmission);
		Emitter(const Emitter&);

		~Emitter();

		SimulationData* getData();
		Vector3	     getWorldPosition();
		unsigned int getMinEmission();
		unsigned int getMaxEmission();
		Vector3	     getVelocityEmission();
		unsigned int getDurationTime();
		unsigned int getCurrentTime();

		void 	     setData(SimulationData*);
		void	     setWorldPosition(Vector3 position);
		void	     setMinEmission(unsigned int minEmission);
		void 	     setMaxEmission(unsigned int maxEmission);
		void	     setVelocityEmission(Vector3 velocityEmission);
		void	     setDurationTime(unsigned int durationTime);
		void	     setCurrentTime(unsigned int currentTime);

		virtual vector<Particle*> emitParticles() = 0;
		virtual void display(Vector3) = 0;

	protected:

		SimulationData *data;
      	        Vector3      worldPosition;
		unsigned int minEmission;
		unsigned int maxEmission;
		Vector3  velocityEmission;
		unsigned int durationTime;
		unsigned int currentTime;

		void   addParticle(Vector3 pos, vector<Particle*> *particles);
};

#endif
