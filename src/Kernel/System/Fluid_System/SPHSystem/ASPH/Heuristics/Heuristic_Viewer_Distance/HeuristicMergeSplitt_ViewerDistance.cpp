#include <HeuristicMergeSplitt_ViewerDistance.h>
#include <HeuristicMergeSplitt.cuh>

/**************************************************************************************/
/**************************************************************************************/
HeuristicMergeSplitt_ViewerDistance::HeuristicMergeSplitt_ViewerDistance(unsigned int nbBodies):HeuristicMergeSplitt(nbBodies)
{
	this->distanceMax = 10000000;
}
/**************************************************************************************/
HeuristicMergeSplitt_ViewerDistance::HeuristicMergeSplitt_ViewerDistance(unsigned int nbBodies, Vector3 viewer, float distanceMax)
				    :HeuristicMergeSplitt(nbBodies)
{
	this->viewer = viewer;
	this->distanceMax = distanceMax;
}
/**************************************************************************************/
HeuristicMergeSplitt_ViewerDistance::HeuristicMergeSplitt_ViewerDistance(const HeuristicMergeSplitt_ViewerDistance& H)
				     :HeuristicMergeSplitt(H)
{
	this->viewer = H.viewer;
	this->distanceMax = H.distanceMax;
}
/**************************************************************************************/
HeuristicMergeSplitt_ViewerDistance::~HeuristicMergeSplitt_ViewerDistance()
{
}
/**************************************************************************************/
/**************************************************************************************/
Viewer HeuristicMergeSplitt_ViewerDistance::getViewer()
{
	return viewer;
}
/**************************************************************************************/
float HeuristicMergeSplitt_ViewerDistance::getDistanceMax()
{
	return distanceMax;
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicMergeSplitt_ViewerDistance::setViewer(Vector3 viewer)
{
	this->viewer = viewer;
}
/**************************************************************************************/
void HeuristicMergeSplitt_ViewerDistance::setDistanceMax(float distanceMax)
{
	this->distanceMax = distanceMax;
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicMergeSplitt_ViewerDistance::evaluateFunction(float* pos, float* vel, float* forces)
{
	evaluateHeuristic_ViewerDistance(pos, viewer, distanceMax, nbBodies, m_dRes);
}
/**************************************************************************************/
/**************************************************************************************/
