#include <AnimatedPeriodicHeightField.h>
#include <AnimatedPeriodicHeightField.cuh>
#include <GL/gl.h>
#include <stdio.h>
/***********************************************************************************/
/***********************************************************************************/
AnimatedPeriodic_HeightField::AnimatedPeriodic_HeightField()
			     :Periodic_HeightField(),AnimatedHeightField()
{
	t = 0;
}
/***********************************************************************************/
AnimatedPeriodic_HeightField::AnimatedPeriodic_HeightField(float step)
			     :Periodic_HeightField(),AnimatedHeightField(step)
{
	t = 0;
}
/***********************************************************************************/
AnimatedPeriodic_HeightField::AnimatedPeriodic_HeightField(const AnimatedPeriodic_HeightField& H)
			     :Periodic_HeightField(H),AnimatedHeightField(H)
{
	this->t = H.t;
}
/***********************************************************************************/
AnimatedPeriodic_HeightField::~AnimatedPeriodic_HeightField()
{
}
/***********************************************************************************/
/***********************************************************************************/
void AnimatedPeriodic_HeightField::create(Vector3 origin, float length, float width, double dx_, double dz_,
			    		  uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
			    		  double phiMin, double phiMax, double wMin, double wMax, double step)
{
	setStep(step);
	Vector3 min_(origin.x()-length/2,origin.y(),origin.z()-width/2);
	Vector3 max_(origin.x()+length/2,origin.y(),origin.z()+width/2);

	Periodic_HeightField::create(nbFunc,AMin,AMax,kMin,kMax,thetaMin,thetaMax,phiMin,phiMax,wMin,wMax);
	HeightField::create(min_,max_,dx_,dz_);
	HeightField::generate();
	t = 0;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void AnimatedPeriodic_HeightField::update()
{
	uint size = nbPosX + nbPosX*nbPosZ;
	AnimatedPeriodicHeightField_calculateHeight_CUDA(m_pos, m_A, m_k, m_theta, m_phi, m_w, t, nbFunc, size);
	copyArrayFromDevice(pos,m_pos, 0, 3*size*sizeof(double));
        threadSync();
	t += step;
}
/***********************************************************************************/
/***********************************************************************************/
void AnimatedPeriodic_HeightField::loadSpectrum(const char* filename)
{
	Periodic_HeightField::loadSpectrum(filename);
	HeightField::create(Min,Max,dx,dz);
	HeightField::generate();
	t = 0;
}
/***********************************************************************************/
/***********************************************************************************/
void AnimatedPeriodic_HeightField::saveSpectrum(const char* filename)
{
	Periodic_HeightField::saveSpectrum(filename);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void AnimatedPeriodic_HeightField::display(Vector3 color)
{
	HeightField::display(color);
}
/*****************************************************************************************************/
void AnimatedPeriodic_HeightField::displayNormales(Vector3 color)
{
	HeightField::displayNormale(color);
}
/*****************************************************************************************************/
/*****************************************************************************************************/

