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
				addParticle(pos,&particles);
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
