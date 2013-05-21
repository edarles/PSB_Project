#include <EmitterSphere.h>
#include <stdio.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>
/************************************************************************************************/
/************************************************************************************************/
EmitterSphere::EmitterSphere():Emitter(),Sphere()
{
}
/************************************************************************************************/
EmitterSphere::EmitterSphere(const Sphere& M, Vector3 velocityEmission)
	    :Emitter(Vector3(0,0,0),1,1,1,velocityEmission),Sphere(M)
{
}
/************************************************************************************************/
EmitterSphere::EmitterSphere(Vector3 center, float baseRadius, Vector3 velocityEmission)
	      :Emitter(Vector3(0,0,0),1,1,1,velocityEmission),Sphere(center,baseRadius)
{
}
/************************************************************************************************/
EmitterSphere::EmitterSphere(const EmitterSphere& M):Emitter(M),Sphere(M)
{
}
/************************************************************************************************/
EmitterSphere::~EmitterSphere()
{
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterSphere::emitParticles()
{
	if(data!=NULL){
	vector<Particle*> particles;
	if(currentTime<durationTime){
		double y = center.y();
		while(y<center.y()+radius){
			double r = 0;
			while(r<=sqrt(radius*radius - y*y)){
				double theta = 0;
				double thetaAp = 1;
				double dTheta = acos(r/sqrt(r*r+data->getParticleRadius()*data->getParticleRadius()));
				while (theta<=2*M_PI && thetaAp>0){	
					double x = center.x() + r*cos(theta);
					double z = center.z() + r*sin(theta);
					Vector3 pos = Vector3(x,y,z);
					theta += dTheta;
					thetaAp = theta;
					if(typeid(*data)==typeid(SimulationData_SimpleSystem)){
						Particle *p = new Particle(pos,velocityEmission, data->getParticleMass(),
									   data->getParticleRadius(), data->getColor());
						particles.push_back(p);
					}
					if(typeid(*data)==typeid(SimulationData_CudaSystem)){
						SimulationData_CudaSystem* dataCuda = (SimulationData_CudaSystem*) data;
						CudaParticle *p = new CudaParticle(pos,velocityEmission,
							   dataCuda->getParticleMass(), dataCuda->getParticleRadius(), dataCuda->getColor(),
						           dataCuda->getInteractionRadius(), dataCuda->getSpring(), 
						           dataCuda->getDamping(), dataCuda->getShear(), dataCuda->getAttraction());
						particles.push_back(p);
					}
					if(typeid(*data)==typeid(SimulationData_SPHSystem)){
						SimulationData_SPHSystem* dataSPH = (SimulationData_SPHSystem*) data;
						SPHParticle *p = new SPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 		dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
							dataSPH->getKernelParticles(), 0,
					 		dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 		dataSPH->getThreshold(), dataSPH->getSurfaceTension(), dataSPH->getViscosity());
						particles.push_back(p);
					}
					if(typeid(*data)==typeid(SimulationData_PCI_SPHSystem)){
						SimulationData_PCI_SPHSystem* dataSPH = (SimulationData_PCI_SPHSystem*) data;
						PCI_SPHParticle *p = new PCI_SPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 		dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(),
							dataSPH->getKernelParticles(), 0,
					 		dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 		dataSPH->getThreshold(), dataSPH->getSurfaceTension(), 
							dataSPH->getViscosity(),dataSPH->getTemperature());
						particles.push_back(p);
					}
				}
				r += data->getParticleRadius();
			}
			y += data->getParticleRadius();
		}
		y = center.y()-data->getParticleRadius();
		while(y>=center.y()-radius){
			double r = 0;
			while(r<=sqrt(radius*radius - y*y)){
				double theta = 0;
				double thetaAp = 1;
				double dTheta = acos(r/sqrt(r*r+data->getParticleRadius()*data->getParticleRadius()));
				while (theta<=2*M_PI && thetaAp>0){	
					double x = center.x() + r*cos(theta);
					double z = center.z() + r*sin(theta);
					Vector3 pos = Vector3(x,y,z);
					theta += dTheta;
					thetaAp = theta;
					if(typeid(*data)==typeid(SimulationData_SimpleSystem)){
						Particle *p = new Particle(pos,velocityEmission, data->getParticleMass(),
									   data->getParticleRadius(), data->getColor());
						particles.push_back(p);
					}
					if(typeid(*data)==typeid(SimulationData_CudaSystem)){
						SimulationData_CudaSystem* dataCuda = (SimulationData_CudaSystem*) data;
						CudaParticle *p = new CudaParticle(pos,velocityEmission,
							   dataCuda->getParticleMass(), dataCuda->getParticleRadius(), dataCuda->getColor(),
						           dataCuda->getInteractionRadius(), dataCuda->getSpring(), 
						           dataCuda->getDamping(), dataCuda->getShear(), dataCuda->getAttraction());
						particles.push_back(p);
					}
					if(typeid(*data)==typeid(SimulationData_SPHSystem)){
						SimulationData_SPHSystem* dataSPH = (SimulationData_SPHSystem*) data;
						SPHParticle *p = new SPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 		dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(), 
							dataSPH->getKernelParticles(), 0,
					 		dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 		dataSPH->getThreshold(), dataSPH->getSurfaceTension(), dataSPH->getViscosity());
						particles.push_back(p);
					}
					if(typeid(*data)==typeid(SimulationData_PCI_SPHSystem)){
						SimulationData_PCI_SPHSystem* dataSPH = (SimulationData_PCI_SPHSystem*) data;
						PCI_SPHParticle *p = new PCI_SPHParticle(pos,velocityEmission,dataSPH->getParticleMass(),
					 		dataSPH->getParticleRadius(), dataSPH->getColor(), dataSPH->getSupportRadius(),
							dataSPH->getKernelParticles(), 0,
					 		dataSPH->getRestDensity(), 0, dataSPH->getGasStiffness(), 
					 		dataSPH->getThreshold(), dataSPH->getSurfaceTension(), 
							dataSPH->getViscosity(),dataSPH->getTemperature());
						particles.push_back(p);
					}
				}
				r += data->getParticleRadius();
			}
			y -= data->getParticleRadius();
		}
		currentTime++;
	}
	return particles;
	}
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterSphere::display(Vector3 color)
{
	//Sphere::display(color);
}
/************************************************************************************************/
/************************************************************************************************/
