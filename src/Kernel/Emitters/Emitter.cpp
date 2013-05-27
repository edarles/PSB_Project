#include <Emitter.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>

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
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles());
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
void   Emitter::addParticle(Vector3 pos, vector<Particle*> *particles)
{
	if(typeid(*data)==typeid(SimulationData_SimpleSystem)){
		Particle *p = new Particle(pos,velocityEmission,
				           data->getParticleMass(),data->getParticleRadius(), data->getColor());
		particles->push_back(p);
	}
	if(typeid(*data)==typeid(SimulationData_CudaSystem)){
		SimulationData_CudaSystem* dataCuda = (SimulationData_CudaSystem*) data;
		CudaParticle *p = new CudaParticle(pos,velocityEmission,
					           dataCuda->getParticleMass(), dataCuda->getParticleRadius(), dataCuda->getColor(),
						   dataCuda->getInteractionRadius(), dataCuda->getSpring(), 
						   dataCuda->getDamping(), dataCuda->getShear(), dataCuda->getAttraction());
		particles->push_back(p);
	}
	if(typeid(*data)==typeid(SimulationData_SPHSystem)){
		SimulationData_SPHSystem* dataSPH = (SimulationData_SPHSystem*) data;
		SPHParticle *p = new SPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
					 dataSPH->getKernelParticles(), 0,
					 dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 dataSPH->getThreshold(), dataSPH->getSurfaceTension(), dataSPH->getViscosity());
		particles->push_back(p);
	}
	if(typeid(*data)==typeid(SimulationData_PCI_SPHSystem)){
		SimulationData_PCI_SPHSystem* dataSPH = (SimulationData_PCI_SPHSystem*) data;
		PCI_SPHParticle *p = new PCI_SPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
					 dataSPH->getKernelParticles(), 0,
					 dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 dataSPH->getThreshold(), dataSPH->getSurfaceTension(), 
					 dataSPH->getViscosity());
		particles->push_back(p);
	}
}
/************************************************************************************************/
