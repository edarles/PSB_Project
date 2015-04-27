#ifndef _BOX_COLLISION_
#define _BOX_COLLISION_

#include <vector>
#include <assert.h>
#include <Box.h>
#include <ObjectCollision.h>
#include <string.h>

using namespace std;
using namespace Utils;

class BoxCollision : public ObjectCollision, public Box {

	public:
		BoxCollision();
		BoxCollision(float elast, bool is_container);
		BoxCollision(const BoxCollision&);

		~BoxCollision();

		void create(Vector3 origin, float length, float width, float depth, float elast, bool is_container);

		void collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             float dt, int nbBodiesP);

		void display(GLenum, GLenum, Vector3 color);
		void displayNormales(Vector3 color);

		void inverseNormales();
};

#endif
