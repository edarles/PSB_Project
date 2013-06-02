#include <ParticleExporter_XML.h>

namespace Utils {
/***********************************************************************/
/***********************************************************************/
ParticleExporter_XML::ParticleExporter_XML():ParticleExporter()
{
}
/***********************************************************************/
ParticleExporter_XML::~ParticleExporter_XML()
{
}
/***********************************************************************/
/***********************************************************************/
void ParticleExporter_XML::_export(const char* filename, System *S)
{
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );

	TiXmlElement * elmtPart = new TiXmlElement( "Particles" );
        doc.LinkEndChild(elmtPart);

	if(typeid(*S) == typeid(SimpleSystem))
		_export_SimpleParticles(doc,elmtPart,(SimpleSystem*)S);

        if(typeid(*S) == typeid(CudaSystem))
		_export_CudaParticles(doc,elmtPart,(CudaSystem*)S);

	if(typeid(*S) == typeid(SPHSystem))
		_export_SPHParticles(doc,elmtPart,(SPHSystem*)S);

	if(typeid(*S) == typeid(PCI_SPHSystem))
		_export_SPHParticles(doc,elmtPart,(PCI_SPHSystem*)S);

	if(typeid(*S) == typeid(MSPHSystem))
		_export_MSPHParticles(doc,elmtPart,(MSPHSystem*)S);

	doc.SaveFile(filename);

}
/***********************************************************************/
/***********************************************************************/
void ParticleExporter_XML::_export_SimpleParticles(TiXmlDocument doc, TiXmlElement *elmtPart, SimpleSystem *S)
{
	elmtPart->SetAttribute("name","Simple System");
	elmtPart->SetDoubleAttribute("dt",S->getDt());
	vector<Particle*> particles = S->getParticles();
	elmtPart->SetAttribute("count",particles.size());
	ForceExt_Constante* F = (ForceExt_Constante*) S->getForcesExt()->getForce(0);
	TiXmlElement * elmtGravity = new TiXmlElement( "Gravity");
	elmtGravity->SetDoubleAttribute("X", F->getDirection().x());
	elmtGravity->SetDoubleAttribute("Y", F->getDirection().y());
	elmtGravity->SetDoubleAttribute("Z", F->getDirection().z());
	elmtGravity->SetDoubleAttribute("Amp", F->getAmplitude());
	elmtPart->LinkEndChild(elmtGravity);

	for(unsigned int i=0;i<particles.size();i++){
			Vector3 pos = Vector3(S->m_hPos[1][i*3],S->m_hPos[1][i*3+1],S->m_hPos[1][i*3+2]);
			Vector3 vel = Vector3(S->m_hVel[1][i*3],S->m_hVel[1][i*3+1],S->m_hVel[1][i*3+2]);
			TiXmlElement * elmtParticle = new TiXmlElement( "Particle");
			elmtParticle->SetAttribute("index", i);
			elmtParticle->SetDoubleAttribute("mass",particles[i]->getMass());
			elmtParticle->SetDoubleAttribute("radius",particles[i]->getParticleRadius());

			TiXmlElement * elmtPos = new TiXmlElement( "position" );
			elmtPos->SetDoubleAttribute("X", pos.x());
			elmtPos->SetDoubleAttribute("Y", pos.y());
			elmtPos->SetDoubleAttribute("Z", pos.z());
			TiXmlElement * elmtVel = new TiXmlElement( "velocity" );
			elmtVel->SetDoubleAttribute("X", vel.x());
			elmtVel->SetDoubleAttribute("Y", vel.y());
			elmtVel->SetDoubleAttribute("Z", vel.z());
			TiXmlElement * elmtColor = new TiXmlElement( "color" );
			elmtColor->SetDoubleAttribute("X", particles[i]->getColor().x());
			elmtColor->SetDoubleAttribute("Y", particles[i]->getColor().y());
			elmtColor->SetDoubleAttribute("Z", particles[i]->getColor().z());
			elmtParticle->LinkEndChild(elmtPos);
			elmtParticle->LinkEndChild(elmtVel);
			elmtParticle->LinkEndChild(elmtColor);
			elmtPart->LinkEndChild(elmtParticle);
	}
}
/***********************************************************************/
/***********************************************************************/
void ParticleExporter_XML::_export_CudaParticles(TiXmlDocument doc, TiXmlElement *elmtPart, CudaSystem *S)
{
	elmtPart->SetAttribute("name","Cuda System");
	elmtPart->SetDoubleAttribute("dt",S->getDt());
	vector<Particle*> particles = S->getParticles();
	elmtPart->SetAttribute("count",particles.size());
	ForceExt_Constante* F = (ForceExt_Constante*) S->getForcesExt()->getForce(0);
	TiXmlElement * elmtGravity = new TiXmlElement( "Gravity");
	elmtGravity->SetDoubleAttribute("X", F->getDirection().x());
	elmtGravity->SetDoubleAttribute("Y", F->getDirection().y());
	elmtGravity->SetDoubleAttribute("Z", F->getDirection().z());
	elmtGravity->SetDoubleAttribute("Amp", F->getAmplitude());
	elmtPart->LinkEndChild(elmtGravity);

	for(unsigned int i=0;i<particles.size();i++){
			CudaParticle* p = (CudaParticle*) particles[i];
			Vector3 pos = Vector3(S->m_hPos[1][i*3],S->m_hPos[1][i*3+1],S->m_hPos[1][i*3+2]);
			Vector3 vel = Vector3(S->m_hVel[1][i*3],S->m_hVel[1][i*3+1],S->m_hVel[1][i*3+2]);
			TiXmlElement * elmtParticle = new TiXmlElement( "Particle");
			elmtParticle->SetAttribute("index", i);
			elmtParticle->SetDoubleAttribute("mass",particles[i]->getMass());
			elmtParticle->SetDoubleAttribute("radius",particles[i]->getParticleRadius());
			elmtParticle->SetDoubleAttribute("interactionRadius",p->getInteractionRadius());
			elmtParticle->SetDoubleAttribute("spring",p->getSpring());
			elmtParticle->SetDoubleAttribute("damping",p->getDamping());
			elmtParticle->SetDoubleAttribute("shear",p->getShear());
			elmtParticle->SetDoubleAttribute("attraction",p->getAttraction());
			
			TiXmlElement * elmtPos = new TiXmlElement( "position" );
			elmtPos->SetDoubleAttribute("X", pos.x());
			elmtPos->SetDoubleAttribute("Y", pos.y());
			elmtPos->SetDoubleAttribute("Z", pos.z());
			TiXmlElement * elmtVel = new TiXmlElement( "velocity" );
			elmtVel->SetDoubleAttribute("X", vel.x());
			elmtVel->SetDoubleAttribute("Y", vel.y());
			elmtVel->SetDoubleAttribute("Z", vel.z());
			TiXmlElement * elmtColor = new TiXmlElement( "color" );
			elmtColor->SetDoubleAttribute("X", particles[i]->getColor().x());
			elmtColor->SetDoubleAttribute("Y", particles[i]->getColor().y());
			elmtColor->SetDoubleAttribute("Z", particles[i]->getColor().z());
			elmtParticle->LinkEndChild(elmtPos);
			elmtParticle->LinkEndChild(elmtVel);
			elmtParticle->LinkEndChild(elmtColor);
			elmtPart->LinkEndChild(elmtParticle);
	}
}
/***********************************************************************/
/***********************************************************************/
void ParticleExporter_XML::_export_SPHParticles(TiXmlDocument doc, TiXmlElement *elmtPart, SPHSystem *S)
{
	elmtPart->SetAttribute("name","SPH System");
	elmtPart->SetDoubleAttribute("dt",S->getDt());
	vector<Particle*> particles = S->getParticles();
	elmtPart->SetAttribute("count",particles.size());
	ForceExt_Constante* F = (ForceExt_Constante*) S->getForcesExt()->getForce(0);
	TiXmlElement * elmtGravity = new TiXmlElement( "Gravity");
	elmtGravity->SetDoubleAttribute("X", F->getDirection().x());
	elmtGravity->SetDoubleAttribute("Y", F->getDirection().y());
	elmtGravity->SetDoubleAttribute("Z", F->getDirection().z());
	elmtGravity->SetDoubleAttribute("Amp", F->getAmplitude());
	elmtPart->LinkEndChild(elmtGravity);

	for(unsigned int i=0;i<particles.size();i++){
			SPHParticle* p = (SPHParticle*) particles[i];
			Vector3 pos = Vector3(S->m_hPos[1][i*3],S->m_hPos[1][i*3+1],S->m_hPos[1][i*3+2]);
			Vector3 vel = Vector3(S->m_hVel[1][i*3],S->m_hVel[1][i*3+1],S->m_hVel[1][i*3+2]);
			Vector3 velInterAv = Vector3(S->m_hVelInterAv[i*3],S->m_hVelInterAv[i*3+1],S->m_hVelInterAv[i*3+2]);
			Vector3 velInterAp = Vector3(S->m_hVelInterAp[i*3],S->m_hVelInterAp[i*3+1],S->m_hVelInterAp[i*3+2]);
			TiXmlElement * elmtParticle = new TiXmlElement( "Particle");
			elmtParticle->SetAttribute("index", i);
			elmtParticle->SetDoubleAttribute("mass",p->getMass());
			elmtParticle->SetDoubleAttribute("radius",p->getParticleRadius());
			elmtParticle->SetDoubleAttribute("kernelParticles",p->getKernelParticles());
			elmtParticle->SetDoubleAttribute("interactionRadius",p->getInteractionRadius());
			elmtParticle->SetDoubleAttribute("restDensity",p->getRestDensity());
			elmtParticle->SetDoubleAttribute("gasStiffness",p->getGasStiffness());
			elmtParticle->SetDoubleAttribute("threshold",p->getThreshold());
			elmtParticle->SetDoubleAttribute("surfaceTension",p->getSurfaceTension());
			elmtParticle->SetDoubleAttribute("viscosity",p->getViscosity());

			TiXmlElement * elmtPos = new TiXmlElement( "position" );
			elmtPos->SetDoubleAttribute("X", pos.x());
			elmtPos->SetDoubleAttribute("Y", pos.y());
			elmtPos->SetDoubleAttribute("Z", pos.z());
			TiXmlElement * elmtVel = new TiXmlElement( "velocity" );
			elmtVel->SetDoubleAttribute("X", vel.x());
			elmtVel->SetDoubleAttribute("Y", vel.y());
			elmtVel->SetDoubleAttribute("Z", vel.z());
			TiXmlElement * elmtVelInterAv = new TiXmlElement( "velocityInterAv" );
			elmtVelInterAv->SetDoubleAttribute("X", velInterAv.x());
			elmtVelInterAv->SetDoubleAttribute("Y", velInterAv.y());
			elmtVelInterAv->SetDoubleAttribute("Z", velInterAv.z());
			TiXmlElement * elmtVelInterAp = new TiXmlElement( "velocityInterAp" );
			elmtVelInterAp->SetDoubleAttribute("X", velInterAp.x());
			elmtVelInterAp->SetDoubleAttribute("Y", velInterAp.y());
			elmtVelInterAp->SetDoubleAttribute("Z", velInterAp.z());
			TiXmlElement * elmtColor = new TiXmlElement( "color" );
			elmtColor->SetDoubleAttribute("X", particles[i]->getColor().x());
			elmtColor->SetDoubleAttribute("Y", particles[i]->getColor().y());
			elmtColor->SetDoubleAttribute("Z", particles[i]->getColor().z());
			elmtParticle->LinkEndChild(elmtPos);
			elmtParticle->LinkEndChild(elmtVel);
			elmtParticle->LinkEndChild(elmtVelInterAv);
			elmtParticle->LinkEndChild(elmtVelInterAp);
			elmtParticle->LinkEndChild(elmtColor);
			elmtPart->LinkEndChild(elmtParticle);
	}
}
/***********************************************************************/
/***********************************************************************/
void ParticleExporter_XML::_export_MSPHParticles(TiXmlDocument doc, TiXmlElement *elmtPart, MSPHSystem *S)
{
	elmtPart->SetAttribute("name","MSPH System");
	elmtPart->SetDoubleAttribute("dt",S->getDt());
	vector<Particle*> particles = S->getParticles();
	elmtPart->SetAttribute("count",particles.size());
	ForceExt_Constante* F = (ForceExt_Constante*) S->getForcesExt()->getForce(0);
	TiXmlElement * elmtGravity = new TiXmlElement( "Gravity");
	elmtGravity->SetDoubleAttribute("X", F->getDirection().x());
	elmtGravity->SetDoubleAttribute("Y", F->getDirection().y());
	elmtGravity->SetDoubleAttribute("Z", F->getDirection().z());
	elmtGravity->SetDoubleAttribute("Amp", F->getAmplitude());
	elmtPart->LinkEndChild(elmtGravity);

	copyArrayFromDevice(S->m_hMass,S->m_dMass,0,sizeof(double)*particles.size());
    	copyArrayFromDevice(S->m_hTemp,S->m_dTemp,0,sizeof(double)*particles.size());
	copyArrayFromDevice(S->m_hRestDensity,S->m_restDensity,0,sizeof(double)*particles.size());

	for(unsigned int i=0;i<particles.size();i++){
			MSPHParticle* p = (MSPHParticle*) particles[i];
			Vector3 pos = Vector3(S->m_hPos[1][i*3],S->m_hPos[1][i*3+1],S->m_hPos[1][i*3+2]);
			Vector3 vel = Vector3(S->m_hVel[1][i*3],S->m_hVel[1][i*3+1],S->m_hVel[1][i*3+2]);
			Vector3 velInterAv = Vector3(S->m_hVelInterAv[i*3],S->m_hVelInterAv[i*3+1],S->m_hVelInterAv[i*3+2]);
			Vector3 velInterAp = Vector3(S->m_hVelInterAp[i*3],S->m_hVelInterAp[i*3+1],S->m_hVelInterAp[i*3+2]);

			double restD = S->m_hRestDensity[i];
			double temp = S->m_hTemp[i];
			double mass = S->m_hMass[i];

			TiXmlElement * elmtParticle = new TiXmlElement( "Particle");
			elmtParticle->SetAttribute("index", i);
			elmtParticle->SetDoubleAttribute("mass",mass);
			elmtParticle->SetDoubleAttribute("radius",p->getParticleRadius());
			elmtParticle->SetDoubleAttribute("kernelParticles",p->getKernelParticles());
			elmtParticle->SetDoubleAttribute("interactionRadius",p->getInteractionRadius());
			elmtParticle->SetDoubleAttribute("restDensity",restD);
			elmtParticle->SetDoubleAttribute("gasStiffness",p->getGasStiffness());
			elmtParticle->SetDoubleAttribute("threshold",p->getThreshold());
			elmtParticle->SetDoubleAttribute("surfaceTension",p->getSurfaceTension());
			elmtParticle->SetDoubleAttribute("viscosity",p->getViscosity());
			elmtParticle->SetDoubleAttribute("temperature",temp);

			TiXmlElement * elmtPos = new TiXmlElement( "position" );
			elmtPos->SetDoubleAttribute("X", pos.x());
			elmtPos->SetDoubleAttribute("Y", pos.y());
			elmtPos->SetDoubleAttribute("Z", pos.z());
			TiXmlElement * elmtVel = new TiXmlElement( "velocity" );
			elmtVel->SetDoubleAttribute("X", vel.x());
			elmtVel->SetDoubleAttribute("Y", vel.y());
			elmtVel->SetDoubleAttribute("Z", vel.z());
			TiXmlElement * elmtVelInterAv = new TiXmlElement( "velocityInterAv" );
			elmtVelInterAv->SetDoubleAttribute("X", velInterAv.x());
			elmtVelInterAv->SetDoubleAttribute("Y", velInterAv.y());
			elmtVelInterAv->SetDoubleAttribute("Z", velInterAv.z());
			TiXmlElement * elmtVelInterAp = new TiXmlElement( "velocityInterAp" );
			elmtVelInterAp->SetDoubleAttribute("X", velInterAp.x());
			elmtVelInterAp->SetDoubleAttribute("Y", velInterAp.y());
			elmtVelInterAp->SetDoubleAttribute("Z", velInterAp.z());
			TiXmlElement * elmtColor = new TiXmlElement( "color" );
			elmtColor->SetDoubleAttribute("X", particles[i]->getColor().x());
			elmtColor->SetDoubleAttribute("Y", particles[i]->getColor().y());
			elmtColor->SetDoubleAttribute("Z", particles[i]->getColor().z());
			elmtParticle->LinkEndChild(elmtPos);
			elmtParticle->LinkEndChild(elmtVel);
			elmtParticle->LinkEndChild(elmtVelInterAv);
			elmtParticle->LinkEndChild(elmtVelInterAp);
			elmtParticle->LinkEndChild(elmtColor);
			elmtPart->LinkEndChild(elmtParticle);
	}
}
/****************************************************************************************************************/
/****************************************************************************************************************/
}
