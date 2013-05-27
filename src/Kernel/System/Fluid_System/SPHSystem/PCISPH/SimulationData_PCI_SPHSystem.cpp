#include <SimulationData_PCI_SPHSystem.h>

/***********************************************************************************************/
/***********************************************************************************************/
SimulationData_PCI_SPHSystem::SimulationData_PCI_SPHSystem():SimulationData_SPHSystem()
{
}
/***********************************************************************************************/
SimulationData_PCI_SPHSystem::SimulationData_PCI_SPHSystem(float particleRadius, float mass, Vector3 color,
					     		   float restDensity, float viscosity, float surfaceTension,
					     		   float gasStiffness, float kernelParticles)
			     :SimulationData_SPHSystem(particleRadius,mass,color,restDensity,viscosity,surfaceTension,gasStiffness,kernelParticles)
{
}
/***********************************************************************************************/
SimulationData_PCI_SPHSystem::SimulationData_PCI_SPHSystem(const SimulationData_PCI_SPHSystem& S):SimulationData_SPHSystem(S)
{
}
/***********************************************************************************************/
SimulationData_PCI_SPHSystem::~SimulationData_PCI_SPHSystem()
{
}
/***********************************************************************************************/
/***********************************************************************************************/





