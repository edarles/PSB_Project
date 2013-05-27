#include <EmitterGirly.h>
#include <stdio.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>
/************************************************************************************************/
/************************************************************************************************/
EmitterGirly::EmitterGirly():Emitter()
{
}
/************************************************************************************************/
EmitterGirly::EmitterGirly(Vector3 velocityEmission)
	    :Emitter(Vector3(0,0,0),1,1,1,velocityEmission)
{
}
/************************************************************************************************/
EmitterGirly::EmitterGirly(const EmitterGirly& M):Emitter(M)
{
}
/************************************************************************************************/
EmitterGirly::~EmitterGirly()
{
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterGirly::emitParticles()
{
	if(data!=NULL){
	vector<Particle*> particles;
	if(currentTime<durationTime){
		double res;
		double eps = 0.001;
		double step = 0.02;
		for(double x = -3.0; x<=3.0; x+=step){
			for(double y = -3.0; y<=3.0; y+=step){
				for(double z=-3.0; z<=3.0; z+=step){
					res = powf((x*x + (9/4)*y*y + z*z -1),3) - x*x*z*z*z - (9/80)*y*y*z*z*z;
					if(res>=-eps & res<=eps){
						Vector3 pos = Vector3(x,y,z);
						addParticle(pos,&particles);
					}
				}
			}
		}
		currentTime++;
	}
	return particles;
	}
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterGirly::addParticle(Vector3 pos, vector<Particle*> *particles){
	if(typeid(*data)==typeid(SimulationData_SimpleSystem)){
		Particle *p = new Particle(pos,velocityEmission, data->getParticleMass(),data->getParticleRadius(), data->getColor());
		particles->push_back(p);
	}
	if(typeid(*data)==typeid(SimulationData_CudaSystem)){
		SimulationData_CudaSystem* dataCuda = (SimulationData_CudaSystem*) data;
		CudaParticle *p = new CudaParticle(pos,velocityEmission,dataCuda->getParticleMass(), dataCuda->getParticleRadius(), 
		dataCuda->getColor(),dataCuda->getInteractionRadius(), dataCuda->getSpring(), 
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
		dataSPH->getViscosity(),dataSPH->getTemperature());
		particles->push_back(p);
	}
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterGirly::display(Vector3 color)
{
	//Sphere::display(color);
}
/************************************************************************************************/
/************************************************************************************************/
