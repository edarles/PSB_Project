#ifndef _COMBINED_HEIGHT_FIELD_COLLISION_
#define _COMBINED_HEIGHT_FIELD_COLLISION_

#include <CombinedHeightField.h>
#include <HeightFieldCollision.h>
#include <vector>

using namespace std;
using namespace Utils;
/*****************************************************************************************************/
/*****************************************************************************************************/
class Combined_HeightFieldCollision : public Combined_HeightField, public HeightFieldCollision
{
/*****************************************************************************************************/
	public:
		Combined_HeightFieldCollision();
		Combined_HeightFieldCollision(float elast);
		Combined_HeightFieldCollision(const Combined_HeightFieldCollision&);
		~Combined_HeightFieldCollision();
/*****************************************************************************************************/
/*****************************************************************************************************/
		void create(Vector3 origin, float length, float width, double dx, double dz, 
			    vector<HeightField*> Hfields, float elast);
/*****************************************************************************************************/
/*****************************************************************************************************/
		void collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             float dt, int nbBodiesP);
/*****************************************************************************************************/
/*****************************************************************************************************/
		void    addHeightField(HeightField* H);
		void    setHeightField(HeightField* H, uint index);
		void    removeHeightFields(uint index);
		void    clearAll();
/*****************************************************************************************************/
		void display(GLenum, GLenum, Vector3 color);
		void displayNormales(Vector3 color);
};
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
