#include <Emitter.h>
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
       // data = new SimulationData_SPHSystem(0.02,0.01,Vector3(0,0,1),998.29,3.5,0.0728,3,20);
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
	if(typeid(*data)==typeid(SimulationData_SPHSystem2D)){
		SimulationData_SPHSystem2D* d = (SimulationData_SPHSystem2D*) data;
		this->data = new SimulationData_SPHSystem2D(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles());
	}
	if(typeid(*data)==typeid(SimulationData_WCSPHSystem)){
		SimulationData_WCSPHSystem* d = (SimulationData_WCSPHSystem*) data;
		this->data = new SimulationData_WCSPHSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles());
	}
	if(typeid(*data)==typeid(SimulationData_PCI_SPHSystem)){
		SimulationData_PCI_SPHSystem* d = (SimulationData_PCI_SPHSystem*) data;
		this->data = new SimulationData_PCI_SPHSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles());
	}
	if(typeid(*data)==typeid(SimulationData_MSPHSystem)){
		SimulationData_MSPHSystem* d = (SimulationData_MSPHSystem*) data;
		this->data = new SimulationData_MSPHSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles(),
		d->getTemperature(),d->getSigma(),d->getBeta(),d->getG());
	}
	if(typeid(*data)==typeid(SimulationData_SWSPHSystem)){
		SimulationData_SWSPHSystem* d = (SimulationData_SWSPHSystem*) data;
		this->data = new SimulationData_SWSPHSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles());
	}
        if(typeid(*data)==typeid(SimulationData_HybridSPHSystem)){
		SimulationData_HybridSPHSystem* d = (SimulationData_HybridSPHSystem*) data;
		this->data = new SimulationData_HybridSPHSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles(),
		d->getSPHDomain(0),d->getSPHDomain(1),d->getSWSPHDomain(0),d->getSWSPHDomain(1));
	}
	if(typeid(*data)==typeid(SimulationData_HSPHSystem)){
		SimulationData_HSPHSystem* d = (SimulationData_HSPHSystem*) data;
		this->data = new SimulationData_HSPHSystem(d->getParticleRadius(),d->getParticleMass(),d->getColor(),
		d->getRestDensity(),d->getViscosity(),d->getSurfaceTension(),d->getGasStiffness(),d->getKernelParticles(),
		d->getLevelMax(),d->getNbChildsMax());
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
	if(typeid(*data)==typeid(SimulationData_SPHSystem2D)){
		SimulationData_SPHSystem2D* dataSPH = (SimulationData_SPHSystem2D*) data;
		SPH2DParticle *p = new SPH2DParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
					 dataSPH->getKernelParticles(), 0,
					 dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 dataSPH->getThreshold(), dataSPH->getSurfaceTension(), dataSPH->getViscosity());
		particles->push_back(p);
	}
	if(typeid(*data)==typeid(SimulationData_WCSPHSystem)){
		SimulationData_WCSPHSystem* dataSPH = (SimulationData_WCSPHSystem*) data;
		WCSPHParticle *p = new WCSPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
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
	if(typeid(*data)==typeid(SimulationData_MSPHSystem)){
		SimulationData_MSPHSystem* dataSPH = (SimulationData_MSPHSystem*) data;
		MSPHParticle *p = new MSPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
					 dataSPH->getKernelParticles(), 0,
					 dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 dataSPH->getThreshold(), dataSPH->getSurfaceTension(), dataSPH->getViscosity(),
					 dataSPH->getTemperature(),dataSPH->getSigma(),dataSPH->getBeta(),dataSPH->getG(), dataSPH->getPhase());
		particles->push_back(p);
	}
	if(typeid(*data)==typeid(SimulationData_HSPHSystem)){
		SimulationData_HSPHSystem* dataSPH = (SimulationData_HSPHSystem*) data;
		HSPHParticle *p = new HSPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
					 dataSPH->getKernelParticles(), 0,
					 dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 dataSPH->getThreshold(), dataSPH->getSurfaceTension(), dataSPH->getViscosity());
		p->setLevel(0);
		p->setNbChildsMax(dataSPH->getNbChildsMax());
		particles->push_back(p);
	}
	if(typeid(*data)==typeid(SimulationData_SWSPHSystem)){
		SimulationData_SWSPHSystem* dataSPH = (SimulationData_SWSPHSystem*) data;
		SWSPHParticle *p = new SWSPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
					 dataSPH->getKernelParticles(), 0,
					 dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 dataSPH->getThreshold(), dataSPH->getSurfaceTension(), dataSPH->getViscosity());
		particles->push_back(p);
	}
	if(typeid(*data)==typeid(SimulationData_HybridSPHSystem)){
		SimulationData_HybridSPHSystem* dataSPH = (SimulationData_HybridSPHSystem*) data;
		Vector3 S0Min = dataSPH->getSPHDomain(0);
		Vector3 S0Max = dataSPH->getSPHDomain(1);
		Vector3 S1Min = dataSPH->getSWSPHDomain(0); 
		Vector3 S1Max = dataSPH->getSWSPHDomain(1);
		//printf("domain SPH:%f %f %f - %f %f %f\n",S0Min.x(),S0Min.y(),S0Min.z(),S0Max.x(),S0Max.y(),S0Max.z());
		//printf("domain SW-SPH:%f %f %f - %f %f %f\n",S1Min.x(),S1Min.y(),S1Min.z(),S1Max.x(),S1Max.y(),S1Max.z());
		if(pos.x()>=S0Min.x() && pos.y()>=S0Min.y() && pos.z()>=S0Min.z() &&
		   pos.x()<=S0Max.x() && pos.y()<=S0Max.y() && pos.z()<=S0Max.z()){
		   //printf("add SPH particles\n");
		   SPHParticle *p = new SPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
					 dataSPH->getKernelParticles(), 0,
					 dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 dataSPH->getThreshold(), dataSPH->getSurfaceTension(), dataSPH->getViscosity());
		   particles->push_back(p);
		}
		else {
		   //printf("pos:%f %f %f\n",pos.x(),pos.y(),pos.z());
		   if(pos.x()>=S1Min.x() && pos.y()>=S1Min.y() && pos.z()>=S1Min.z() &&
		      pos.x()<=S1Max.x() && pos.y()<=S1Max.y() && pos.z()<=S1Max.z()){
		   	//printf("add Shallow SPH particles\n");
		   	SWSPHParticle *p = new SWSPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
					 dataSPH->getKernelParticles(), 0,
					 dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 dataSPH->getThreshold(), dataSPH->getSurfaceTension(), dataSPH->getViscosity());
		   	particles->push_back(p);
		   }
		}
	}
}
/************************************************************************************************/
