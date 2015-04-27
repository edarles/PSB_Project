#ifndef _CYLINDER_COLLISION_
#define _CYLINDER_COLLISION_

#include <vector>
#include <assert.h>
#include <Cylinder.h>
#include <ObjectCollision.h>
#include <string.h>

using namespace std;
using namespace Utils;

/************************************************************************************/
/************************************************************************************/
class CylinderCollision : public ObjectCollision, public Cylinder {

	public:
	/************************************************************************************/
		CylinderCollision();
		CylinderCollision(float elast, bool is_container);
		CylinderCollision(const CylinderCollision&);

		~CylinderCollision();

	/************************************************************************************/
	/************************************************************************************/
		void create(Vector3 center, float baseRadius, float length,   Vector3 direction, float elast, bool is_container);

	/************************************************************************************/
	/************************************************************************************/
		void collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             float dt, int nbBodiesP);

	/************************************************************************************/
	/************************************************************************************/
		void display(GLenum, GLenum, Vector3 color);
		void displayNormales(Vector3 color);
	/************************************************************************************/
	/************************************************************************************/
		void inverseNormales();
};
/************************************************************************************/
/************************************************************************************/
#endif
