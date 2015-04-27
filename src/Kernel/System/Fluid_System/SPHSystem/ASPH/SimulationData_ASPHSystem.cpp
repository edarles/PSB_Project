#include <SimulationData_ASPHSystem.h>
/**********************************************************************************/
/**********************************************************************************/
SimulationData_ASPHSystem::SimulationData_ASPHSystem():SimulationData_SPHSystem()
{
	nbChildrenMax = 1000;
}
/**********************************************************************************/
SimulationData_ASPHSystem::SimulationData_ASPHSystem(float particleRadius, float mass,      float deltaTime, 
					             float restDensity,    float viscosity, float surfaceTension,
					             float gasStiffness,   float kernelParticles, unsigned int nbChildrenMax)
{
	this->nbChildrenMax = nbChildrenMax;
}
/**********************************************************************************/
SimulationData_ASPHSystem::~SimulationData_ASPHSystem()
{
}
/**********************************************************************************/
/**********************************************************************************/
unsigned int SimulationData_ASPHSystem::getNbChildrenMax()
{
	return nbChildrenMax;
}
/**********************************************************************************/
/**********************************************************************************/
void SimulationData_ASPHSystem::setNbChildrenMax(unsigned int nbChildrenMax)
{
	this->nbChildrenMax = nbChildrenMax;
}
/**********************************************************************************/
/**********************************************************************************/
bool SimulationData_ASPHSystem::loadConfiguration(const char* filename)
{
  TiXmlDocument doc(filename);
  if (doc.LoadFile()){
	TiXmlHandle hDoc(&doc);
	TiXmlElement* elem;
	TiXmlElement* hRoot = doc.FirstChildElement( "Fluid_Material_ASPH" );
	if(hRoot){
		hRoot->QueryFloatAttribute("visualizationRadius",&this->particleRadius);
		hRoot->QueryFloatAttribute("particle_mass",&this->particleMass);
		hRoot->QueryFloatAttribute("deltaTime",&this->deltaTime);
		hRoot->QueryFloatAttribute("restDensity",&this->restDensity);
		hRoot->QueryFloatAttribute("viscosity",&this->viscosity);
		hRoot->QueryFloatAttribute("surfaceTension",&this->surfaceTension);
		hRoot->QueryFloatAttribute("threshold",&this->threshold);
		hRoot->QueryFloatAttribute("gasStiffness",&this->gasStiffness);
		hRoot->QueryFloatAttribute("kernelParticles",&this->kernelParticles);
		hRoot->QueryFloatAttribute("supportRadius",&this->supportRadius);
		hRoot->QueryIntAttribute("Nb_Children_per_particle",&this->nbChildrenMax);
		return true;
	}
	return false;
    }
    else
	return false;
}
/**********************************************************************************/
bool SimulationData_ASPHSystem::saveConfiguration(const char* filename)
{
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );
	TiXmlElement * elmt = new TiXmlElement( "Fluid_Material_ASPH" );
        doc.LinkEndChild(elmt);
	elmt->SetAttribute("name","Fluid_Material");
	elmt->SetDoubleAttribute("visualizationRadius",this->particleRadius);
	elmt->SetDoubleAttribute("particle_mass",this->particleMass);
	elmt->SetDoubleAttribute("deltaTime",this->deltaTime);
	elmt->SetDoubleAttribute("restDensity",this->restDensity);
	elmt->SetDoubleAttribute("viscosity",this->viscosity);
	elmt->SetDoubleAttribute("surfaceTension",this->surfaceTension);
	elmt->SetDoubleAttribute("threshold",this->threshold);
	elmt->SetDoubleAttribute("gasStiffness",this->gasStiffness);
	elmt->SetDoubleAttribute("kernelParticles",this->kernelParticles);
	elmt->SetDoubleAttribute("supportRadius",this->supportRadius);
	elmt->SetAttribute("Nb_Children_per_particle",this->nbChildrenMax);
	doc.SaveFile(filename);
	return true;
}
/**********************************************************************************/
/**********************************************************************************/
