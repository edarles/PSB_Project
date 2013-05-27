#include <EmitterMesh.h>
#include <stdio.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>
/************************************************************************************************/
/************************************************************************************************/
EmitterMesh::EmitterMesh():Emitter(),Mesh()
{
}
/************************************************************************************************/
EmitterMesh::EmitterMesh(const Mesh& M, unsigned int minEmission, 
			 unsigned int maxEmission, unsigned int durationTime, Vector3 velocityEmission)
	    :Emitter(Vector3(0,0,0),minEmission,maxEmission,durationTime,velocityEmission),Mesh(M)
{
}
/************************************************************************************************/
EmitterMesh::EmitterMesh(const EmitterMesh& M):Emitter(M),Mesh(M)
{
}
/************************************************************************************************/
EmitterMesh::~EmitterMesh()
{
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterMesh::emitParticles()
{
	if(data!=NULL){
	vector<Particle*> particles;
	if(currentTime<durationTime){
		srand(time(NULL)); 
		for(unsigned int j=0;j<getNbFaces();j++){
			Vector3 A = getFace(j).getVertex(0);
			Vector3 B = getFace(j).getVertex(1);
			Vector3 C = getFace(j).getVertex(2);
			for(unsigned int i=0;i<maxEmission;i++){
				float a =  (rand() * 0.5 / RAND_MAX);
				float b =  (rand() * 0.5 / RAND_MAX);
				float c =  (rand() * 0.5 / RAND_MAX);
				Vector3 pos = (a*A + b*B + c*C)/(a+b+c);
			
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
		}
		currentTime++;
	}
	return particles;
	}
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterMesh::display(Vector3 color)
{
	Mesh::display(color);
}
/************************************************************************************************/
/************************************************************************************************/
