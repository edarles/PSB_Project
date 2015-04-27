#ifndef _ASPH_PARTICLE_
#define _ASPH_PARTICLE_

#include <SphParticle.h>
#include <vector.h>

using namespace std;

/**************************************************************************************/
/**************************************************************************************/
class ASPHParticle : public SPHParticle 
{
	public:
/**************************************************************************************/
		ASPHParticle();
		ASPHParticle(Vector3 pos, Vector3 vel, double mass);
		ASPHParticle(Vector3 pos, Vector3 vel, double mass, float density, float pressure, 
			     float densityRest, float interactionRadius, float viscosity);
		ASPHParticle(const ASPHParticle&);

		~ASPHParticle();

/**************************************************************************************/
/**************************************************************************************/
		bool getActive();
		bool getCanBeMerged();
		bool getCanBeSplitted();
		vector<ASPHParticle*> getChildren();
		ASPHParticle* getChild(unsigned int i);
		unsigned int getNbChildren();

/**************************************************************************************/
/**************************************************************************************/
		void setActive(bool);
		void setCanBeMerged(bool);
		void setCanBeSplitted(bool);
		void setChildren(vector<ASPHParticle>);
		void setChild(unsigned int i, ASPHParticle*);

/**************************************************************************************/
/**************************************************************************************/
		void addChild(ASPHParticle*);
		void removeChild();
		void removeChild(unsigned int i);

	private:
/**************************************************************************************/
		bool active;
		bool canBeMerged, canBeSplitted;
		vector<ASPHParticle*> children;
/**************************************************************************************/
};
/**************************************************************************************/
/**************************************************************************************/
#endif
