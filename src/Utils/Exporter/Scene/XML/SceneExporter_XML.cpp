#include <SceneExporter_XML.h>
#include <typeinfo>

namespace Utils {

SceneExporter_XML::SceneExporter_XML():SceneExporter()
{
}
SceneExporter_XML::~SceneExporter_XML()
{
}
void SceneExporter_XML::_export(const char* filename, System *S)
{
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );

	TiXmlElement * elmtSystem = new TiXmlElement( "System" );
        doc.LinkEndChild( elmtSystem );

	TiXmlElement * elmtPart = new TiXmlElement( "Particles" );
        elmntSystem.LinkEndChild( elmtPart );

	if(typeid(*S) == typeid(SimpleSystem))
		_export_SimpleParticles(doc,elmtPart,S);

        if(typeid(*S) == typeid(CudaSystem))
		_export_CudaParticles(doc,elmtPart,S);

	TiXmlElement * elmtForce = new TiXmlElement( "Forces" );
        TiXmlElement * elmtEmitter = new TiXmlElement( "Emitters" );

	TiXmlElement * elmtCollision = new TiXmlElement( "Collisions" );
	elmtSystem.LinkEndChild(elmtCollision);
	for(unsigned int i=0;i<S->getObjectsCollision.size();i++){
		ObjectCollision* O = S->getObjectCollision(i);
		if(typeid(*O) == typeid(BoxCollision)) {
			BoxCollision *B = (BoxCollision*) O;
			TiXmlElement * elmtBox = new TiXmlElement( "Box" );
			elmtCollision.LinkEndChild(elmtBox);
			elmtBox->SetDoubleAttribute("restitution",B->getElast());
			elmtBox->SetDoubleAttribute("friction",B->getFriction());
			if(B->getIsContainer())
				elmtBox->SetAttribute("container",1);
			else
				elmtBox->SetAttribute("container",0);
			TiXmlElement * elmtCenterBox = new TiXmlElement( "center" );
			elmtCenterBox->SetDoubleAttribute("x",B->getCenter().x());
			elmtCenterBox->SetDoubleAttribute("y",B->getCenter().y());
			elmtCenterBox->SetDoubleAttribute("z",B->getCenter().z());
			elmtBox.LinkEndChild(elmtCenterBox);
			elmtBox->SetDoubleAttribute("sizeX",B->getSizeX());
			elmtBox->SetDoubleAttribute("sizeY",B->getSizeY());
			elmtBox->SetDoubleAttribute("sizeZ",B->getSizeZ());
		}
		if(typeid(*O) == typeid(SphereCollision)) {
			SphereCollision *S = (SphereCollision*) O;
			TiXmlElement * elmtSphere = new TiXmlElement( "Sphere" );
			elmtCollision.LinkEndChild(elmtBox);
			elmtSphere->SetDoubleAttribute("restitution",S->getElast());
			elmtSphere->SetDoubleAttribute("friction",S->getFriction());
			if(S->getIsContainer())
				elmtSphere->SetAttribute("container",1);
			else
				elmtSphere->SetAttribute("container",0);
			TiXmlElement * elmtCenterSphere = new TiXmlElement( "center" );
			elmtCenterSphere->SetDoubleAttribute("x",S->getCenter().x());
			elmtCenterSphere->SetDoubleAttribute("y",S->getCenter().y());
			elmtCenterSphere->SetDoubleAttribute("z",S->getCenter().z());
			elmtSphere.LinkEndChild(elmtCenterSphere);
			elmtSphere->SetDoubleAttribute("sizeX",S->getRadius());
		}
	}
	doc.SaveFile(filename);

}

