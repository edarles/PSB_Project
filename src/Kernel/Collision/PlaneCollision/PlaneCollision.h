#ifndef _PLANE_COLLISION_
#define _PLANE_COLLISION_

#include <Quadrilateral.h>
#include <ObjectCollision.h>

using namespace Utils;

class PlaneCollision : public ObjectCollision, public Quadrilateral {

	public:
		PlaneCollision();
		PlaneCollision(float elast, float friction, bool is_container);
		PlaneCollision(const PlaneCollision&);

		~PlaneCollision();

		void create(Vector3 origin, float length, float width, float elast, float friction, bool is_container, Vector3 direction);

		void collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             float dt, int nbBodiesP);

		void display(GLenum, GLenum, Vector3 color);
		void displayNormales(Vector3 color);

};

#endif
