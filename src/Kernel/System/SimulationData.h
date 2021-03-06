#ifndef SIMULATION_DATA_
#define SIMULATION_DATA_

#include <Vector3.h>
#include <tinyxml.h>

class SimulationData {

	public:

		SimulationData();
		SimulationData(float particleRadius, float particleMass, Vector3 color);
		SimulationData(const SimulationData&);
		virtual ~SimulationData();

		float 	  getParticleRadius();
		float	  getParticleMass();
		Vector3   getColor();

		void 	  setParticleRadius(float);
		void      setParticleMass(float);
		void      setColor(Vector3 color);

		virtual bool loadConfiguration(const char* filename)=0;
		virtual bool saveConfiguration(const char* filename)=0;

	protected:

    		float   particleRadius;
		float   particleMass;
		Vector3 color;
};

#endif
