#ifndef SIMULATION_DATA_MSPH_SYSTEM
#define SIMULATION_DATA_MSPH_SYSTEM

#include <SimulationData_SPHSystem.h>
/**********************************************************************************/
/**********************************************************************************/
class SimulationData_MSPHSystem : public SimulationData_SPHSystem {

	public:

/**********************************************************************************/
/**********************************************************************************/
		SimulationData_MSPHSystem();
		SimulationData_MSPHSystem(float particleRadius, float mass, Vector3 color,
					  float restDensity, float viscosity, float surfaceTension,
					  float gasStiffness, float kernelParticles, 
					  float temperature,Vector3 sigma, Vector3 beta, Vector3 g);

		SimulationData_MSPHSystem(const SimulationData_MSPHSystem&);

		~SimulationData_MSPHSystem();

/**********************************************************************************/
/**********************************************************************************/
		float getTemperature();
		Vector3 getSigma();
		Vector3 getBeta();
		Vector3 getG();
/**********************************************************************************/
/**********************************************************************************/
		void  setTemperature(float temperature);
		void  setSigma(Vector3 sigma);
		void  setBeta(Vector3 beta);
		void  setG(Vector3 g);
/**********************************************************************************/
/**********************************************************************************/
		virtual bool loadConfiguration(const char* filename);
		virtual bool saveConfiguration(const char* filename);

	protected:
/**********************************************************************************/
/**********************************************************************************/
		float temperature;
		Vector3 sigma, beta, g;
/**********************************************************************************/
/**********************************************************************************/
};
/**********************************************************************************/
/**********************************************************************************/
#endif
