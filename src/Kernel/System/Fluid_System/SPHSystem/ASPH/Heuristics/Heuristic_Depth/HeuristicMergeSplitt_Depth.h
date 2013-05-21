#ifndef _HEURISTIC_MERGE_SPLITT_DEPTH_
#define _HEURISTIC_MERGE_SPLITT_DEPTH_

#include <HeuristicMergeSplitt.h>

/**************************************************************************************/
/**************************************************************************************/
class HeuristicMergeSplitt_Depth : public HeuristicMergeSplitt
{
	public:
/**************************************************************************************/
		HeuristicMergeSplitt_Depth(unsigned int nbBodies);
		HeuristicMergeSplitt_Depth(unsigned int nbBodies, float depthMax);
		HeuristicMergeSplitt_Depth(const HeuristicMergeSplitt_Depth&);
		~HeuristicMergeSplitt_Depth();

/**************************************************************************************/
/**************************************************************************************/
		float getDepthMax();

/**************************************************************************************/
/**************************************************************************************/
		void  setDepthMax(float);

/**************************************************************************************/
/**************************************************************************************/
		void evaluateFunction(float* pos, float* vel, float *forces);

	private:
/**************************************************************************************/
		float depthMax; 
/**************************************************************************************/
};

/**************************************************************************************/
/**************************************************************************************/
#endif