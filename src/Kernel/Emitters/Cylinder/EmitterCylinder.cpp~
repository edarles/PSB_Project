#include <EmitterCylinder.h>
#include <stdio.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>
/************************************************************************************************/
/************************************************************************************************/
EmitterCylinder::EmitterCylinder():Emitter(),Cylinder()
{
}
/************************************************************************************************/
EmitterCylinder::EmitterCylinder(const Cylinder& M, Vector3 velocityEmission)
	    :Emitter(Vector3(0,0,0),1,1,1,velocityEmission),Cylinder(M)
{
}
/************************************************************************************************/
EmitterCylinder::EmitterCylinder(Vector3 center, float baseRadius, float length, Vector3 direction, Vector3 velocityEmission)
		:Emitter(Vector3(0,0,0),1,1,1,velocityEmission),Cylinder(center,baseRadius,length,direction)
{
}
/************************************************************************************************/
EmitterCylinder::EmitterCylinder(const EmitterCylinder& M):Emitter(M),Cylinder(M)
{
}
/************************************************************************************************/
EmitterCylinder::~EmitterCylinder()
{
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterCylinder::emitParticles()
{
	if(data!=NULL){
        printf("emitParticles\n");
	vector<Particle*> particles;
	if(currentTime<durationTime){
		Vector3 P1 = Vector3(center.x() + direction.x()*(length/2),center.y() + direction.y()*(length/2),center.z() + direction.z()*(length/2));
		Vector3 P2 = Vector3(center.x() - direction.x()*(length/2),center.y() - direction.y()*(length/2),center.z() - direction.z()*(length/2));
		double y = P2.y();
		//Vector3 dir = Vector3(1-direction.x,1-direction.y,1-direction.z)
		while(y<P1.y()){
			double r = 0;
			while(r<=baseRadius){
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
		currentTime++;
	}
	return particles;
	}
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterCylinder::display(Vector3 color)
{
	Cylinder::display(color);
}
/************************************************************************************************/
/************************************************************************************************/
