#include <HeuristicMergeSplitt_ForceMagnitude.h>
#include <HeuristicMergeSplitt.cuh>

/**************************************************************************************/
/**************************************************************************************/
HeuristicMergeSplitt_ForceMagnitude::HeuristicMergeSplitt_ForceMagnitude(unsigned int nbBodies):HeuristicMergeSplitt(nbBodies)
{
	this->magnitudeMax = 10000000;
}
/**************************************************************************************/
HeuristicMergeSplitt_ForceMagnitude::HeuristicMergeSplitt_ForceMagnitude(unsigned int nbBodies, float magnitudeMax)
				     :HeuristicMergeSplitt(nbBodies)
{
	this->magnitudeMax = magnitudeMax;
}
/**************************************************************************************/
HeuristicMergeSplitt_ForceMagnitude::HeuristicMergeSplitt_ForceMagnitude(const HeuristicMergeSplitt_ForceMagnitude& H)
				    :HeuristicMergeSplitt(H)
{
	this->magnitudeMax = H.magnitudeMax;
}
/**************************************************************************************/
HeuristicMergeSplitt_ForceMagnitude::~HeuristicMergeSplitt_ForceMagnitude()
{
}
/**************************************************************************************/
/**************************************************************************************/
float HeuristicMergeSplitt_ForceMagnitude::getMagnitudeMax()
{
	return magnitudeMax;
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicMergeSplitt_ForceMagnitude::setMagnitudeMax(float magnitudeMax)
{
	this->magnitudeMax = magnitudeMax;
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicMergeSplitt_ForceMagnitude::evaluateFunction(float* pos, float* vel, float* forces)
{
	evaluateHeuristic_ForceMagnitude(forces, magnitudeMax, nbBodies, m_dRes);
}
/**************************************************************************************/
/**************************************************************************************/
