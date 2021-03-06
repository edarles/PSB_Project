#ifndef _COLLISION_
#define _COLLISION_

#define MAX_FACES 10000
#define MAX_OBJECTS 100

#include <vector>
#include <assert.h>
#include <ObjectCollision.h>

using namespace std;

class Collision {

	public:
		Collision();
		~Collision();

		ObjectCollision* getObject(unsigned int);
		vector<ObjectCollision*> getObjects();

		void setObject(unsigned int, ObjectCollision*);
		void setObject(ObjectCollision*);
		void removeLastObject();

		void collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             float dt, int nbBodiesP);

		void display(GLenum modeRaster, GLenum modeFace, Vector3 colorObject, Vector3 colorNormales, bool normales);


	private:
		vector<ObjectCollision*> objects;

};

#endif
