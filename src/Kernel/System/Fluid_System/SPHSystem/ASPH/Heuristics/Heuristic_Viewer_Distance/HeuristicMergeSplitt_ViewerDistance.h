#ifndef _HEURISTIC_MERGE_SPLITT_VIEWER_DISTANCE_
#define _HEURISTIC_MERGE_SPLITT_VIEWER_DISTANCE_

#include <HeuristicMergeSplitt.h>
#include <Vector3.h>

/**************************************************************************************/
/**************************************************************************************/
class HeuristicMergeSplitt_ViewerDistance : public HeuristicMergeSplitt
{
	public:
/**************************************************************************************/
		HeuristicMergeSplitt_ViewerDistance(unsigned int nbBodies);
		HeuristicMergeSplitt_ViewerDistance(unsigned int nbBodies, Vector3 viewer, float distanceMax);
		HeuristicMergeSplitt_ViewerDistance(const HeuristicMergeSplitt_ViewerDistance&);
		~HeuristicMergeSplitt_ViewerDistance();

/**************************************************************************************/
/**************************************************************************************/
		Vector3 getViewer();
		float   getDistanceMax();

/**************************************************************************************/
/**************************************************************************************/
		void   setViewer(Vector3);
		void   setDistanceMax(float);

/**************************************************************************************/
/**************************************************************************************/
		void evaluateFunction(float* pos, float* vel, float* forces);

	private:
/**************************************************************************************/
		Vector3 viewer;
		float   distanceMax;
/**************************************************************************************/
};

/**************************************************************************************/
/**************************************************************************************/
#endif
