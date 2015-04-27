#ifndef PERIODIC_HEIGHT_FIELD_COLLISION_
#define PERIODIC_HEIGHT_FIELD_COLLISION_

#include <PeriodicHeightField.h>
#include <HeightFieldCollision.h>

using namespace Utils;
/***************************************************************************************************************/
/***************************************************************************************************************/
class Periodic_HeightFieldCollision : public Periodic_HeightField, public HeightFieldCollision
{
/***************************************************************************************************************/
	public:
		Periodic_HeightFieldCollision();
		Periodic_HeightFieldCollision(float elast);
		Periodic_HeightFieldCollision(const Periodic_HeightFieldCollision&);
		~Periodic_HeightFieldCollision();
/***************************************************************************************************************/
		void create(Vector3 origin, float length, float width, double dx, double dz,
			    uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
			    double phiMin, double phiMax, double wMin, double wMax, float elast);
/***************************************************************************************************************/
		void collide(double* oldPos, double* newPos, double* oldVel, double* newVel, float radiusParticle, 
		             float dt, int nbBodiesP);
/***************************************************************************************************************/
		void saveSpectrum(const char* filename);
		void loadSpectrum(const char* filename);
/*****************************************************************************************************/
		void display(GLenum, GLenum, Vector3 color);
		void displayNormales(Vector3 color);
};
/***************************************************************************************************************/
/***************************************************************************************************************/
#endif
/***************************************************************************************************************/
/***************************************************************************************************************/
