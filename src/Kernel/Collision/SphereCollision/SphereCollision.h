#ifndef _SPHERE_COLLISION_
#define _SPHERE_COLLISION_

#include <ObjectCollision.h>
#include <Sphere.h>

using namespace Utils;

class SphereCollision : public ObjectCollision, public Sphere {

	public:
		SphereCollision();
		SphereCollision(float elast, float friction, bool is_container);
		SphereCollision(const SphereCollision&);
		SphereCollision(Vector3 origin, float radius, float elast, float friction, bool is_container);

		~SphereCollision();

		void collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             float dt, int nbBodiesP);

		void display(GLenum mode, GLenum raster, Vector3 color);
		void displayNormales(Vector3 color);

};

#endif
