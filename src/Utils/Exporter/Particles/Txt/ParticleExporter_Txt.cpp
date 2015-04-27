#include <ParticleExporter_Txt.h>
#include <stdio.h>
#include <typeinfo>

namespace Utils {
/*******************************************************************************/
/*******************************************************************************/
ParticleExporter_Txt::ParticleExporter_Txt():ParticleExporter()
{
}
/*******************************************************************************/
ParticleExporter_Txt::~ParticleExporter_Txt()
{
}
/*******************************************************************************/
/*******************************************************************************/
void ParticleExporter_Txt::_exportToMaya(unsigned int frame, const char* directory, System *S)
{
 char buffer[100];
 double var = 0.0;
 sprintf(buffer, "%s/positions%d.txt", directory,frame);
 FILE *f1 = fopen(buffer,"w");
 sprintf(buffer, "%s/ids%d.txt", directory,frame);
 FILE *f2 = fopen(buffer,"w");
 sprintf(buffer, "%s/nb%d.txt", directory,frame);
 FILE *f3 = fopen(buffer,"w");
 if(f1!=NULL && f2!=NULL && f3!=NULL & S!=NULL){
 	vector<Particle*> particles = S->getParticles();
 	for (unsigned int i = 0; i < particles.size(); i++) {
		Particle *p = particles[i];
		fprintf(f1,"%f %f %f\n",S->m_hPos[1][i*3],S->m_hPos[1][i*3+1],S->m_hPos[1][i*3+2]);
		fprintf(f2,"%f\n",var);
		var = var + 1.0;
	 }
 	fprintf(f3,"%d\n",particles.size());
 	fclose(f1);
 	fclose(f2);
 	fclose(f3);
 }
}
/*******************************************************************************/
/*******************************************************************************/
void ParticleExporter_Txt::_export(const char* filename, System *S)
{
 FILE *f = fopen(filename,"w");
 if(f!=NULL && S!=NULL){
	if(typeid(*S) == typeid(SimpleSystem)){
		vector<Particle*> particles = S->getParticles();
		fprintf(f,"%d\n",particles.size());
		for(unsigned int i=0;i<particles.size();i++){
			Particle *p = particles[i];
			fprintf(f,"particle -p %f %f %f -v %f %f %f -m %f\n",S->m_hPos[1][i*3],S->m_hPos[1][i*3+1],S->m_hPos[1][i*3+2],S->m_hVel[1][i*3],S->m_hVel[1][i*3+1],S->m_hVel[1][i*3+2],p->getMass());
		}
		fclose(f);
	}
	if(typeid(*S) == typeid(SPHSystem)){
		SPHSystem *system = (SPHSystem*) S;
		unsigned int count = system->getParticles().size();
		fprintf(f,"SPH %d\n",count);
		for(unsigned int i=0;i<count;i++){
			fprintf(f,"-p %lf %lf %lf -m %lf -d %lf\n", system->m_hPos[1][i*3], system->m_hPos[1][i*3+1], system->m_hPos[1][i*3+2], system->m_hMass[i], system->m_hDensity[i]);
		}
		fclose(f);
	}
  }
}
/*******************************************************************************/
/*******************************************************************************/
}
/*******************************************************************************/
/*******************************************************************************/
