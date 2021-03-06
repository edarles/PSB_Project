#ifndef SIMULATION_DATA_PCI_SPH_SYSTEM
#define SIMULATION_DATA_PCI_SPH_SYSTEM

#include <SimulationData_SPHSystem.h>
/**********************************************************************************/
/**********************************************************************************/
class SimulationData_PCI_SPHSystem : public SimulationData_SPHSystem {

	public:

/**********************************************************************************/
/**********************************************************************************/
		SimulationData_PCI_SPHSystem();
		SimulationData_PCI_SPHSystem(float particleRadius, float mass, Vector3 color,
					     float restDensity, float viscosity, float surfaceTension,
					     float gasStiffness, float kernelParticles);

		SimulationData_PCI_SPHSystem(const SimulationData_PCI_SPHSystem&);

		~SimulationData_PCI_SPHSystem();
/**********************************************************************************/
/**********************************************************************************/
};
/**********************************************************************************/
/**********************************************************************************/
#endif
