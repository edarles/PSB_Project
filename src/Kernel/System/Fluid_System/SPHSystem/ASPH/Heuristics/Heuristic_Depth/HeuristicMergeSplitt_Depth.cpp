#include <HeuristicMergeSplitt_Depth.h>
#include <HeuristicMergeSplitt.cuh>

/**************************************************************************************/
/**************************************************************************************/
HeuristicMergeSplitt_Depth::HeuristicMergeSplitt_Depth(unsigned int nbBodies):HeuristicMergeSplitt(nbBodies)
{
	this->depthMax = -10000000;
}
/**************************************************************************************/
HeuristicMergeSplitt_Depth::HeuristicMergeSplitt_Depth(float depthMax, unsigned int nbBodies):HeuristicMergeSplitt(nbBodies)
{
	this->depthMax = depthMax;
}
/**************************************************************************************/
HeuristicMergeSplitt_Depth::HeuristicMergeSplitt_Depth(const HeuristicMergeSplitt_Depth& H):HeuristicMergeSplitt(H)
{
	this->depthMax = H.depthMax;
}
/**************************************************************************************/
HeuristicMergeSplitt_Depth::~HeuristicMergeSplitt_Depth()
{
}
/**************************************************************************************/
/**************************************************************************************/
float HeuristicMergeSplitt_Depth::getDepthMax()
{
	return depthMax;
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicMergeSplitt_Depth::setDepthMax(float depthMax)
{
	this->depthMax = depthMax;
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicMergeSplitt_Depth::evaluateFunction(float* pos, float* vel, float* forces)
{
	evaluateHeuristic_Depth(pos,depthMax,nbBodies,m_dRes);
}
/**************************************************************************************/
/**************************************************************************************/
