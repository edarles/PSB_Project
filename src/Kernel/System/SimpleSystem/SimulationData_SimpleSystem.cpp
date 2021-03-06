#include "SimulationData_SimpleSystem.h"

/**************************************************************************************************************************/
/**************************************************************************************************************************/
SimulationData_SimpleSystem::SimulationData_SimpleSystem():SimulationData()
{
}
/**************************************************************************************************************************/
SimulationData_SimpleSystem::SimulationData_SimpleSystem(float particleRadius, float mass, Vector3 color)
			    :SimulationData(particleRadius, mass, color)
{
}
/**************************************************************************************************************************/
SimulationData_SimpleSystem::SimulationData_SimpleSystem(const SimulationData_SimpleSystem& S)
			    :SimulationData(S)
{
}
/**************************************************************************************************************************/
SimulationData_SimpleSystem::~SimulationData_SimpleSystem()
{
}
/**************************************************************************************************************************/
/**************************************************************************************************************************/
bool SimulationData_SimpleSystem::loadConfiguration(const char* filename)
{
  TiXmlDocument doc(filename);
  if (doc.LoadFile()){

	TiXmlHandle hDoc(&doc);
	TiXmlElement* elem;
	TiXmlElement* hRoot = doc.FirstChildElement( "Simple_System_Material" );
        elem=hDoc.FirstChildElement().Element();	
	hRoot->QueryFloatAttribute("visualizationRadius",&this->particleRadius);
	hRoot->QueryFloatAttribute("particle_mass",&this->particleMass);
	return true;
    }
    else
	return false;
}
/**************************************************************************************************************************/
/**************************************************************************************************************************/
bool SimulationData_SimpleSystem::saveConfiguration(const char* filename)
{
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );
	TiXmlElement * elmt = new TiXmlElement( "Simple_System_Material" );
        doc.LinkEndChild(elmt);
	elmt->SetAttribute("name","Simple_System_Material");
	elmt->SetAttribute("visualizationRadius",this->particleRadius);
	elmt->SetAttribute("particle_mass",this->particleMass);
	//elmt->SetAttribute("particle_color",this->particleMass);
	doc.SaveFile(filename);
	return true;
}
/**************************************************************************************************************************/
/**************************************************************************************************************************/
