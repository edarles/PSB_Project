#ifndef SIMULATION_DATA_SPH_SYSTEM
#define SIMULATION_DATA_SPH_SYSTEM

#include <Vector3.h>
#include <SimulationData.h>
#include <stdlib.h>

/**********************************************************************************/
/**********************************************************************************/
class SimulationData_SPHSystem : public SimulationData {

	public:

/**********************************************************************************/
/**********************************************************************************/
		SimulationData_SPHSystem();
		SimulationData_SPHSystem(float particleRadius, float mass, Vector3 color,
					 float restDensity, float viscosity, float surfaceTension,
					 float gasStiffness, float kernelParticles);

		SimulationData_SPHSystem(const SimulationData_SPHSystem&);

		~SimulationData_SPHSystem();

/**********************************************************************************/
/**********************************************************************************/
		float getRestDensity();
		float getViscosity();
		float getSurfaceTension();
		float getThreshold();
		float getGasStiffness();
		float getKernelParticles();
		float getSupportRadius();

/**********************************************************************************/
/**********************************************************************************/
		void  setRestDensity(float);
		void  setViscosity(float);
		void  setSurfaceTension(float);
		void  setThreshold(float);
		void  setGasStiffness(float);
		void  setKernelParticles(float);
		void  setSupportRadius(float);

/**********************************************************************************/
/**********************************************************************************/
		virtual bool loadConfiguration(const char* filename);
		virtual bool saveConfiguration(const char* filename);

	protected:
/**********************************************************************************/
/**********************************************************************************/
		float restDensity;
		float viscosity;
		float surfaceTension;
		float threshold;
		float gasStiffness;
		float kernelParticles;
		float supportRadius;
/**********************************************************************************/
/**********************************************************************************/
};
/**********************************************************************************/
/**********************************************************************************/
#endif
