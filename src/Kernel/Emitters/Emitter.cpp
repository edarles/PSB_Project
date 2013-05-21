#include <Emitter.h>
#include <typeinfo>

/************************************************************************************************/
/************************************************************************************************/
Emitter::Emitter()
{
	worldPosition = Vector3(0,0,0);
	minEmission = 0;
	maxEmission = 1000;
	velocityEmission = Vector3(0,0,0);
	durationTime = 0;
	currentTime = 0;
	data = NULL;
}
/************************************************************************************************/
Emitter::Emitter(Vector3 worldPosition,unsigned int minEmission, unsigned int maxEmission, 
		 unsigned int durationTime, Vector3 velocityEmission)
{
	this->worldPosition = worldPosition;
	this->minEmission = minEmission;
	this->maxEmission = maxEmission;
	this->durationTime = durationTime;
	this->velocityEmission = velocityEmission;
	this->currentTime = 0;
	this->data = NULL;
}
/************************************************************************************************/
Emitter::Emitter(const Emitter& E)
{
	this->worldPosition = worldPosition;
	this->minEmission = E.minEmission;
	this->maxEmission = E.maxEmission;
	this->durationTime = E.durationTime;
	this->velocityEmission = E.velocityEmission;
	this->currentTime = E.currentTime;
	this->data = NULL;
}
/************************************************************************************************/
Emitter::~Emitter()
{
	if(data) delete(data);
}
/************************************************************************************************/
/************************************************************************************************/
SimulationData* Emitter::getData()
{
	return data;
}
/************************************************************************************************/
Vector3	Emitter::getWorldPosition()
{
	return worldPosition;
}
/************************************************************************************************/
unsigned int Emitter::getMinEmission()
{
	return minEmission;
}
/************************************************************************************************/
unsigned int Emitter::getMaxEmission()
{
	return maxEmission;
}
/************************************************************************************************/
unsigned int Emitter::getDurationTime()
{
	return durationTime;
}
/************************************************************************************************/
unsigned int Emitter::getCurrentTime()
{
	return currentTime;
}
/************************************************************************************************/
Vector3	Emitter::getVelocityEmission()
{
	return velocityEmission;
}
/************************************************************************************************/
/************************************************************************************************/
void    Emitter::setData(SimulationData* data)
{
	if(this->data!=NULL)  delete(this->data);

	if(typeid(*data)==typeid(SimulationData_SimpleSystem)){
		SimulationData_SimpleSystem* d = (SimulationData_SimpleSystem*) data;
		this->data = new SimulationData_SimpleSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor());
	}
	if(typeid(*data)==typeid(SimulationData_CudaSystem)){
		SimulationData_CudaSystem* d = (SimulationData_CudaSystem*) data;
		this->data = new SimulationData_CudaSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getInteractionRadius(),d->getSpring(),d->getDamping(),d->getShear(),d->getAttraction());
	}
	if(typeid(*data)==typeid(SimulationData_SPHSystem)){
		SimulationData_SPHSystem* d = (SimulationData_SPHSystem*) data;
		this->data = new SimulationData_SPHSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles());
	}
	if(typeid(*data)==typeid(SimulationData_PCI_SPHSystem)){
		SimulationData_PCI_SPHSystem* d = (SimulationData_PCI_SPHSystem*) data;
		this->data = new SimulationData_PCI_SPHSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles(),d->getTemperature());
	}
}
/************************************************************************************************/
void	Emitter::setWorldPosition(Vector3 worldPosition)
{
	this->worldPosition = worldPosition;
}
/************************************************************************************************/
void	Emitter::setMinEmission(unsigned int minEmission)
{
	this->minEmission = minEmission;
}
/************************************************************************************************/
void 	Emitter::setMaxEmission(unsigned int maxEmission)
{
	this->maxEmission = maxEmission;
}
/************************************************************************************************/
void	Emitter::setDurationTime(unsigned int durationTime)
{
	this->durationTime = durationTime;
}
/************************************************************************************************/
void	Emitter::setCurrentTime(unsigned int currentTime)
{
	this->currentTime = currentTime;
}
/************************************************************************************************/
void	Emitter::setVelocityEmission(Vector3 velocityEmission)
{
	this->velocityEmission = velocityEmission;
}
/************************************************************************************************/
/************************************************************************************************/
