#include "SimulationData_CudaSystem.h"

/**********************************************************************************/
/**********************************************************************************/
SimulationData_CudaSystem::SimulationData_CudaSystem():SimulationData_SimpleSystem()
{
}
/**********************************************************************************/
SimulationData_CudaSystem::SimulationData_CudaSystem(float particleRadius, float mass, Vector3 color,
						     float interactionRadius, float spring, float damping, float shear, float attraction)
			  :SimulationData_SimpleSystem(particleRadius,mass, color)
{
	this->interactionRadius = interactionRadius;
	this->spring = spring;
	this->damping = damping;
	this->shear = shear;
	this->attraction = attraction;
}
/**********************************************************************************/
SimulationData_CudaSystem::SimulationData_CudaSystem(const SimulationData_CudaSystem& S)
			  :SimulationData_SimpleSystem(S)
{
}
/**********************************************************************************/
SimulationData_CudaSystem::~SimulationData_CudaSystem()
{
}
/**********************************************************************************/
/**********************************************************************************/
float SimulationData_CudaSystem::getInteractionRadius()
{
	return interactionRadius;
}
/**********************************************************************************/
float SimulationData_CudaSystem::getSpring()
{
	return spring;
}
/**********************************************************************************/
float SimulationData_CudaSystem::getDamping()
{
	return damping;
}
/**********************************************************************************/
float SimulationData_CudaSystem::getShear()
{
	return shear;
}
/**********************************************************************************/
float SimulationData_CudaSystem::getAttraction()
{
	return attraction;
}
/**********************************************************************************/
/**********************************************************************************/
void  SimulationData_CudaSystem::setInteractionRadius(float interactionRadius)
{
	this->interactionRadius = interactionRadius;
}
/**********************************************************************************/
void  SimulationData_CudaSystem::setSpring(float spring)
{
	this->spring = spring;
}
/**********************************************************************************/
void  SimulationData_CudaSystem::setDamping(float damping)
{
	this->damping = damping;
}
/**********************************************************************************/
void  SimulationData_CudaSystem::setShear(float shear)
{
	this->shear = shear;
}
/**********************************************************************************/
void  SimulationData_CudaSystem::setAttraction(float attraction)
{
	this->attraction = attraction;
}
/**********************************************************************************/
/**********************************************************************************/
bool SimulationData_CudaSystem::loadConfiguration(const char* filename)
{
  TiXmlDocument doc(filename);
  if (doc.LoadFile()){
	TiXmlHandle hDoc(&doc);
	TiXmlElement* elem;
	TiXmlElement* hRoot = doc.FirstChildElement( "Cuda_System_Material" );
        elem=hDoc.FirstChildElement().Element();	
	hRoot->QueryFloatAttribute("visualizationRadius",&this->particleRadius);
	hRoot->QueryFloatAttribute("particle_mass",&this->particleMass);
	hRoot->QueryFloatAttribute("interactionRadius",&this->interactionRadius);
	hRoot->QueryFloatAttribute("spring",&this->spring);
	hRoot->QueryFloatAttribute("damping",&this->damping);
	hRoot->QueryFloatAttribute("shear",&this->shear);
	hRoot->QueryFloatAttribute("attraction",&this->attraction);
	return true;
    }
    else
	return false;
}
/**********************************************************************************/
bool SimulationData_CudaSystem::saveConfiguration(const char* filename)
{
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );
	TiXmlElement * elmt = new TiXmlElement( "Cuda_System_Material" );
        doc.LinkEndChild(elmt);
	elmt->SetAttribute("name","Cuda_System_Material");
	elmt->SetDoubleAttribute("visualizationRadius",this->particleRadius);
	elmt->SetDoubleAttribute("particle_mass",this->particleMass);
	elmt->SetDoubleAttribute("interactionRadius",this->interactionRadius);
	elmt->SetDoubleAttribute("spring",this->spring);
	elmt->SetDoubleAttribute("damping",this->damping);
	elmt->SetDoubleAttribute("shear",this->shear);
	elmt->SetDoubleAttribute("attraction",this->attraction);
	doc.SaveFile(filename);
	return true;
}
/**********************************************************************************/
/**********************************************************************************/
