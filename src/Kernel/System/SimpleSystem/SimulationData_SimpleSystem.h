#ifndef SIMULATION_DATA_SIMPLE_SYSTEM
#define SIMULATION_DATA_SIMPLE_SYSTEM

#include <Vector3.h>
#include <SimulationData.h>

class SimulationData_SimpleSystem : public SimulationData {

	public:

		SimulationData_SimpleSystem();
		SimulationData_SimpleSystem(float particleRadius, float mass, Vector3 color);
		SimulationData_SimpleSystem(const SimulationData_SimpleSystem& S);

		~SimulationData_SimpleSystem();

		bool loadConfiguration(const char* filename);
		bool saveConfiguration(const char* filename);

};

#endif
