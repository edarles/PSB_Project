#include <SimulationData_MSPHSystem.h>

/***********************************************************************************************/
/***********************************************************************************************/
SimulationData_MSPHSystem::SimulationData_PCI_SPHSystem():SimulationData_SPHSystem()
{
	// By default: ambiant temperature (C°)
	temperature = 20;
	// By default: Milk (lowfat)
	sigma = Vector3(0.9126,1.0748,1.2500);
	beta = Vector3(0.9124,1.0744,1.2492);
	g = Vector3(0.932,0.902,0.859);
}
/***********************************************************************************************/
SimulationData_MSPHSystem::SimulationData_PCI_SPHSystem(float particleRadius, float mass, Vector3 color,
					     		   float restDensity, float viscosity, float surfaceTension,
					     		   float gasStiffness, float kernelParticles, 
							   float temperature,Vector3 sigma, Vector3 beta, Vector3 g)

			     :SimulationData_SPHSystem(particleRadius,mass,color,restDensity,viscosity,surfaceTension,
						       gasStiffness,kernelParticles)
{
	this->temperature = temperature;
	this->sigma = sigma;
	this->beta = beta;
	this->g = g;
}
/***********************************************************************************************/
SimulationData_MSPHSystem::SimulationData_MSPHSystem(const SimulationData_MSPHSystem& S)
			  :SimulationData_SPHSystem(S)
{
	this->temperature = S.temperature;
	this->sigma = S.sigma;
	this->beta = S.beta;
	this->g = S.g;
}
/***********************************************************************************************/
SimulationData_MSPHSystem::~SimulationData_MSPHSystem()
{
}
/***********************************************************************************************/
/***********************************************************************************************/
float SimulationData_MSPHSystem::getTemperature()
{
	return temperature;
}
/***********************************************************************************************/
Vector3 SimulationData_MSPHSystem::getSigma()
{
	return sigma;
}
/***********************************************************************************************/
Vector3 SimulationData_MSPHSystem::getBeta()
{
	return beta;
}
/***********************************************************************************************/
Vector3 SimulationData_MSPHSystem::getG()
{
	return g;
}
/***********************************************************************************************/
/***********************************************************************************************/
void SimulationData_MSPHSystem::setTemperature(float temperature)
{
	this->temperature = temperature;
}
/***********************************************************************************************/
void SimulationData_MSPHSystem::setSigma(Vector3 sigma)
{
	this->sigma = sigma;
}
/***********************************************************************************************/
void SimulationData_MSPHSystem::setBeta(Vector3 beta)
{
	this->beta = beta;
}
/***********************************************************************************************/
void SimulationData_MSPHSystem::setG(Vector3 g)
{
	this->g = g;
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
		double sx,sy,sz,bx,by,bz,gx,gy,gz;
		TiXmlElement* sig = hRoot->FirstChildElement("sigma");
		sig->QueryDoubleAttribute("X", &sx);
		sig->QueryDoubleAttribute("Y", &sy);
		sig->QueryDoubleAttribute("Z", &sz);
		TiXmlElement* bet = hRoot->FirstChildElement("beta");
		bet->QueryDoubleAttribute("X", &bx);
		bet->QueryDoubleAttribute("Y", &by);
		bet->QueryDoubleAttribute("Z", &bz);
		TiXmlElement* gX = hRoot->FirstChildElement("G");
		gX->QueryDoubleAttribute("X", &gx);
		gX->QueryDoubleAttribute("Y", &gy);
		gX->QueryDoubleAttribute("Z", &gz);
		sigma = Vector3(sx,sy,sz);
		beta = Vector3(bx,by,bz);
		g = Vector3(gx,gy,gz);
		delete(sig); 
		delete(bet);
		delete(gX);
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
	TiXmlElement * elmtSigma = new TiXmlElement( "sigma" );
	elmtSigma->SetDoubleAttribute("R", sigma.x());
	elmtSigma->SetDoubleAttribute("G", sigma.y());
	elmtSigma->SetDoubleAttribute("B", sigma.z());
	elmt->LinkEndChild(elmtSigma);
	TiXmlElement * elmtBeta = new TiXmlElement( "beta" );
	elmtBeta->SetDoubleAttribute("R", beta.x());
	elmtBeta->SetDoubleAttribute("G", beta.y());
	elmtBeta->SetDoubleAttribute("B", beta.z());
	elmt->LinkEndChild(elmtBeta);
	TiXmlElement * elmtG = new TiXmlElement( "G" );
	elmtG->SetDoubleAttribute("R", g.x());
	elmtG->SetDoubleAttribute("G", g.y());
	elmtG->SetDoubleAttribute("B", g.z());
	elmt->LinkEndChild(elmtG);
	doc.SaveFile(filename);
	delete(elmtSigma); 
	delete(elmtBeta);
	delete(elmtG);
	return true;
}
/***********************************************************************************************/
/***********************************************************************************************/





