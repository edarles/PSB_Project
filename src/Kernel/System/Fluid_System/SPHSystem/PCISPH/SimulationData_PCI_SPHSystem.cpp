#include <SimulationData_PCI_SPHSystem.h>

/***********************************************************************************************/
/***********************************************************************************************/
SimulationData_PCI_SPHSystem::SimulationData_PCI_SPHSystem():SimulationData_SPHSystem()
{
	temperature = 0;
}
/***********************************************************************************************/
SimulationData_PCI_SPHSystem::SimulationData_PCI_SPHSystem(float particleRadius, float mass, Vector3 color,
					     		   float restDensity, float viscosity, float surfaceTension,
					     		   float gasStiffness, float kernelParticles, float temperature)
			     :SimulationData_SPHSystem(particleRadius,mass,color,restDensity,viscosity,surfaceTension,gasStiffness,kernelParticles)
{
	this->temperature = temperature;
}
/***********************************************************************************************/
SimulationData_PCI_SPHSystem::SimulationData_PCI_SPHSystem(const SimulationData_PCI_SPHSystem& S):SimulationData_SPHSystem(S)
{
	this->temperature = S.temperature;
}
/***********************************************************************************************/
SimulationData_PCI_SPHSystem::~SimulationData_PCI_SPHSystem()
{
}
/***********************************************************************************************/
/***********************************************************************************************/
float SimulationData_PCI_SPHSystem::getTemperature()
{
	return temperature;
}
/***********************************************************************************************/
void SimulationData_PCI_SPHSystem::setTemperature(float temperature)
{
	this->temperature = temperature;
}
/***********************************************************************************************/
/***********************************************************************************************/
bool SimulationData_PCI_SPHSystem::loadConfiguration(const char* filename)
{
  TiXmlDocument doc(filename);
  if (doc.LoadFile()){

	TiXmlHandle hDoc(&doc);
	TiXmlElement* elem;
	TiXmlElement* hRoot = doc.FirstChildElement( "Fluid_Material" );
	if(hRoot){
		hRoot->QueryFloatAttribute("visualizationRadius",&this->particleRadius);
		hRoot->QueryFloatAttribute("particle_mass",&this->particleMass);
		hRoot->QueryFloatAttribute("restDensity",&this->restDensity);
		hRoot->QueryFloatAttribute("viscosity",&this->viscosity);
		hRoot->QueryFloatAttribute("surfaceTension",&this->surfaceTension);
		hRoot->QueryFloatAttribute("threshold",&this->threshold);
		hRoot->QueryFloatAttribute("gasStiffness",&this->gasStiffness);
		hRoot->QueryFloatAttribute("kernelParticles",&this->kernelParticles);
		hRoot->QueryFloatAttribute("supportRadius",&this->supportRadius);
		hRoot->QueryFloatAttribute("temperature",&this->temperature);
		return true;
	}
	return false;
    }
    else
	return false;
}
/***********************************************************************************************/
bool SimulationData_PCI_SPHSystem::saveConfiguration(const char* filename)
{
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );
	TiXmlElement * elmt = new TiXmlElement( "Fluid_Material" );
        doc.LinkEndChild(elmt);
	elmt->SetAttribute("name","Fluid_Material");
	elmt->SetDoubleAttribute("visualizationRadius",this->particleRadius);
	elmt->SetDoubleAttribute("particle_mass",this->particleMass);
	elmt->SetDoubleAttribute("restDensity",this->restDensity);
	elmt->SetDoubleAttribute("viscosity",this->viscosity);
	elmt->SetDoubleAttribute("surfaceTension",this->surfaceTension);
	elmt->SetDoubleAttribute("threshold",this->threshold);
	elmt->SetDoubleAttribute("gasStiffness",this->gasStiffness);
	elmt->SetDoubleAttribute("kernelParticles",this->kernelParticles);
	elmt->SetDoubleAttribute("supportRadius",this->supportRadius);
	elmt->SetDoubleAttribute("temperature",this->temperature);
	doc.SaveFile(filename);
	return true;
}
/***********************************************************************************************/
/***********************************************************************************************/





