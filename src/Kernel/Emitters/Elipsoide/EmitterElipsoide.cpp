#include <EmitterElipsoide.h>
#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>

/************************************************************************************************/
/************************************************************************************************/
EmitterElipsoide::EmitterElipsoide():Emitter()
{
	center = Vector3(0,0,0);
	sizeX = 1;
	sizeZ = 1;
}
/************************************************************************************************/
EmitterElipsoide::EmitterElipsoide(Vector3 center, float sizeX, float sizeZ, unsigned int minEmission, 
				   unsigned int maxEmission, unsigned int durationTime, Vector3 velocityEmission)
		 :Emitter(Vector3(0,0,0),minEmission,maxEmission,durationTime,velocityEmission)
{
	this->center = center;
	this->sizeX = sizeX;
	this->sizeZ = sizeZ;
}
/************************************************************************************************/
EmitterElipsoide::EmitterElipsoide(const EmitterElipsoide& E): Emitter(E)
{
	this->center = E.center;
	this->sizeX = E.sizeX;
	this->sizeZ = E.sizeZ;
}
/************************************************************************************************/
EmitterElipsoide::~EmitterElipsoide()
{
}
/************************************************************************************************/
/************************************************************************************************/
Vector3 EmitterElipsoide::getCenter()
{
	return center;
}
/************************************************************************************************/

float   EmitterElipsoide::getSizeX()
{
	return sizeX;
}
/************************************************************************************************/

float   EmitterElipsoide::getSizeZ()
{
	return sizeZ;
}
/************************************************************************************************/
/************************************************************************************************/
void	EmitterElipsoide::setCenter(Vector3 center)
{
	this->center = center;
}
/************************************************************************************************/
void	EmitterElipsoide::setSizeX(float sizeX)
{
	this->sizeX = sizeX;
}
/************************************************************************************************/
void	EmitterElipsoide::setSizeZ(float sizeZ)
{
	this->sizeZ = sizeZ;
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterElipsoide::emitParticles()
{
	vector<Particle*> particles;
	for(unsigned int i=0;i<maxEmission;i++){
		double t = (double)(rand()*2*M_PI/RAND_MAX);
 		Vector3 C(sizeX*cos(t),0,sizeZ*sin(t));
 		float x = (float)(-sizeX/2+rand()*sizeX/RAND_MAX);
		float z = (float)(-sizeZ/2+rand()*sizeZ/RAND_MAX);
		Vector3 pos(center.x()+x,center.y(),center.z()+z);
		pos.setX(pos.x()+C.x()/100);
		pos.setZ(pos.z()+C.z()/100);

		if(typeid(*data)==typeid(SimulationData_SimpleSystem)){
			Particle *p = new Particle(pos,velocityEmission,
						   data->getParticleMass(),data->getParticleRadius(), data->getColor());
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
        currentTime++;
	return particles;
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterElipsoide::display(Vector3 color)
{
	glColor3f(color.x(),color.y(),color.z());
	glColor3f(1,1,1);
	glRotatef(90,1,0,0);
	glScalef(sizeX,1,sizeZ);
	glTranslatef(center.x(),center.y(),center.z());
	GLUquadric *qobj = gluNewQuadric();
	gluDisk(qobj,sizeX,sizeZ,20,20);
	glScalef(-sizeX,-1,-sizeZ);
	glTranslatef(-center.x(),-center.y(),-center.z());
        glRotatef(90,-1,0,0);
 	glColor3f(1,1,1);
}

