#include <ParticleExporter_XML.h>
#include <typeinfo>
#include <CudaParticle.h>

namespace Utils {

ParticleExporter_XML::ParticleExporter_XML():ParticleExporter()
{
}
ParticleExporter_XML::~ParticleExporter_XML()
{
}
void ParticleExporter_XML::_export(const char* filename, System *S)
{
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );

	TiXmlElement * elmtPart = new TiXmlElement( "Particles" );
        doc.LinkEndChild( elmtPart );

	if(typeid(*S) == typeid(SimpleSystem))
		_export_SimpleParticles(doc,elmtPart,S);

        if(typeid(*S) == typeid(CudaSystem))
		_export_CudaParticles(doc,elmtPart,S);

	doc.SaveFile(filename);

}

void ParticleExporter_XML::_export_SimpleParticles(TiXmlDocument doc, TiXmlElement *elmtPart, System *S)
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
			Vector3 pos = particles[i]->getNewPos();
			Vector3 vel = particles[i]->getNewVel();
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
			elmtVel->SetDoubleAttribute("X", particles[i]->getColor().x());
			elmtVel->SetDoubleAttribute("Y", particles[i]->getColor().y());
			elmtVel->SetDoubleAttribute("Z", particles[i]->getColor().z());
			elmtParticle->LinkEndChild(elmtPos);
			elmtParticle->LinkEndChild(elmtVel);
			elmtParticle->LinkEndChild(elmtColor);
			elmtPart->LinkEndChild(elmtParticle);
	}
}
void ParticleExporter_XML::_export_CudaParticles(TiXmlDocument doc, TiXmlElement *elmtPart, System *S)
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
			Vector3 pos = particles[i]->getNewPos();
			Vector3 vel = particles[i]->getNewVel();
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
			elmtVel->SetDoubleAttribute("X", particles[i]->getColor().x());
			elmtVel->SetDoubleAttribute("Y", particles[i]->getColor().y());
			elmtVel->SetDoubleAttribute("Z", particles[i]->getColor().z());
			elmtParticle->LinkEndChild(elmtPos);
			elmtParticle->LinkEndChild(elmtVel);
			elmtParticle->LinkEndChild(elmtColor);
			elmtPart->LinkEndChild(elmtParticle);
	}
}

}