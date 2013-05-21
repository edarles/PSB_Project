#include <ParticleLoader_XML.h>
#include <tinyxml.h>

using namespace std;

namespace Utils {

ParticleLoader_XML::ParticleLoader_XML():ParticleLoader()
{
}
ParticleLoader_XML::~ParticleLoader_XML()
{
}
vector<Particle*> ParticleLoader_XML::load(const char *filename)
{
  vector<Particle*> particles;
  TiXmlDocument doc(filename);
  if (!doc.LoadFile()) return particles;
  else {
	TiXmlHandle hDoc(&doc);
	TiXmlElement* elem;
	TiXmlElement* hRoot = doc.FirstChildElement( "Particles" );
        elem=hDoc.FirstChildElement().Element();
	if(string(elem->Attribute("name"))=="Simple System"){
		TiXmlElement* partElement = hRoot->FirstChildElement("Particle" );
		for( partElement; partElement; partElement=partElement->NextSiblingElement())
		{
			TiXmlElement* position = partElement->FirstChildElement("position");
			double x,y,z;
			position->QueryDoubleAttribute("X", &x);
			position->QueryDoubleAttribute("Y", &y);
			position->QueryDoubleAttribute("Z", &z);
			TiXmlElement* vel = partElement->FirstChildElement("velocity");
			double vx,vy,vz;
			vel->QueryDoubleAttribute("X", &vx);
			vel->QueryDoubleAttribute("Y", &vy);
			vel->QueryDoubleAttribute("Z", &vz);
			double mass, radius;
			partElement->QueryDoubleAttribute("mass",&mass); 
			partElement->QueryDoubleAttribute("radius",&radius); 
			double cx,cy,cz;
			TiXmlElement* color = partElement->FirstChildElement("color");
			color->QueryDoubleAttribute("X", &cx);
			color->QueryDoubleAttribute("Y", &cy);
			color->QueryDoubleAttribute("Z", &cz);
			Particle *p = new Particle(Vector3(x,y,z),Vector3(vx,vy,vz),mass,radius,Vector3(cx,cy,cz));
			particles.push_back(p);
		}
	}
   }
   return particles;
 }

}
