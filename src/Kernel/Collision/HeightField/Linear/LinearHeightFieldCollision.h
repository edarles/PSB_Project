#ifndef _LINEAR_HEIGHT_FIELD_COLLISION_
#define _LINEAR_HEIGHT_FIELD_COLLISION_

#include <LinearHeightField.h>
#include <HeightFieldCollision.h>

using namespace std;
using namespace Utils;
/*****************************************************************************************************/
/*****************************************************************************************************/
class Linear_HeightFieldCollision : public Linear_HeightField, public HeightFieldCollision {

/*****************************************************************************************************/
	public:
		Linear_HeightFieldCollision();
		Linear_HeightFieldCollision(float elast);
		Linear_HeightFieldCollision(const Linear_HeightFieldCollision&);
		~Linear_HeightFieldCollision();
/*****************************************************************************************************/
		void create(Vector3 origin, float length, float width, double dx, double dz, 
			    double a, double b, float elast);
/*****************************************************************************************************/
		void collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             float dt, int nbBodiesP);
/*****************************************************************************************************/
		void display(GLenum, GLenum, Vector3 color);
		void displayNormales(Vector3 color);
};
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
