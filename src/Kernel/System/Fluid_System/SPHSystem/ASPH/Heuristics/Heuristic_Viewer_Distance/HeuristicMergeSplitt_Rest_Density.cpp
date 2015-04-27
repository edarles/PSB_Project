#include <HeuristicMergeSplitt_Rest_Density.h>
#include <HeuristicMergeSplitt.cuh>

/**************************************************************************************/
/**************************************************************************************/
HeuristicMergeSplitt_Rest_Density::HeuristicMergeSplitt_Rest_Density(unsigned int nbBodies):HeuristicMergeSplitt(nbBodies)
{
	this->restDensityMax = 0;
}
/**************************************************************************************/
HeuristicMergeSplitt_Rest_Density::HeuristicMergeSplitt_Rest_Density(unsigned int nbBodies, float restDensityMax)
				  :HeuristicMergeSplitt(nbBodies)
{
	this->restDensityMax = restDensityMax;
}
/**************************************************************************************/
HeuristicMergeSplitt_Rest_Density::HeuristicMergeSplitt_Rest_Density(const HeuristicMergeSplitt_Rest_Density& H)
				  :HeuristicMergeSplitt(H)
{
	this->restDensityMax = H.restDensityMax;
}
/**************************************************************************************/
HeuristicMergeSplitt_Rest_Density::~HeuristicMergeSplitt_Rest_Density()
{
}
/**************************************************************************************/
/**************************************************************************************/
float HeuristicMergeSplitt_Rest_Density::getRestDensityMax()
{
	return restDensityMax;
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicMergeSplitt_Rest_Density::setRestDensityMax(float restDensityMax)
{
	this->restDensityMax = restDensityMax;
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicMergeSplitt_Rest_Density::evaluateFunction(float* pos, float* vel, float* forces, float* currentRestDensity)
{
	evaluateHeuristic_Rest_Density(currentRestDensity, restDensityMax, nbBodies, m_dRes);
}
/**************************************************************************************/
/**************************************************************************************/
