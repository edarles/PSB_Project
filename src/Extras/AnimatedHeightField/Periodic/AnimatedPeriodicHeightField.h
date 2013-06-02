#ifndef ANIMATED_PERIODIC_HEIGHT_FIELD_COLLISION_
#define ANIMATED_PERIODIC_HEIGHT_FIELD_COLLISION_

#include <PeriodicHeightField.h>
#include <AnimatedHeightField.h>

using namespace Utils;
/***************************************************************************************************************/
/***************************************************************************************************************/
class AnimatedPeriodic_HeightField : public Periodic_HeightField, public AnimatedHeightField
{
/***************************************************************************************************************/
	public:
		AnimatedPeriodic_HeightField();
		AnimatedPeriodic_HeightField(float step);
		AnimatedPeriodic_HeightField(const AnimatedPeriodic_HeightField&);
		virtual ~AnimatedPeriodic_HeightField();
/***************************************************************************************************************/
		void create(Vector3 origin, float length, float width, double dx, double dz,
			    uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
			    double phiMin, double phiMax, double wMin, double wMax, double step);
/***************************************************************************************************************/
		void update();
/***************************************************************************************************************/
		void saveSpectrum(const char* filename);
		void loadSpectrum(const char* filename);
/***************************************************************************************************************/
		void display(Vector3);
		void displayNormales(Vector3 color);
};
/***************************************************************************************************************/
/***************************************************************************************************************/
#endif
/***************************************************************************************************************/
/***************************************************************************************************************/
