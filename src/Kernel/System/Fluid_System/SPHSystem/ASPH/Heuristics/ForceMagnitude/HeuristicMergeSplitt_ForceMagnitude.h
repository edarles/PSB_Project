#ifndef _HEURISTIC_MERGE_SPLITT_FORCE_MAGNITUDE_
#define _HEURISTIC_MERGE_SPLITT_FORCE_MAGNITUDE_

#include <HeuristicMergeSplitt.h>

/**************************************************************************************/
/**************************************************************************************/
class HeuristicMergeSplitt_ForceMagnitude : public HeuristicMergeSplitt
{
	public:
/**************************************************************************************/
		HeuristicMergeSplitt_ForceMagnitude(unsigned int nbBodies);
		HeuristicMergeSplitt_ForceMagnitude(unsigned int nbBodies, float magnitudeMax);
		HeuristicMergeSplitt_ForceMagnitude(const HeuristicMergeSplitt_ForceMagnitude&);
		~HeuristicMergeSplitt_ForceMagnitude();

/**************************************************************************************/
/**************************************************************************************/
		float   getMagnitudeMax();
/**************************************************************************************/
/**************************************************************************************/
		void   setMagnitudeMax(float);
/**************************************************************************************/
/**************************************************************************************/
		void evaluateFunction(float* pos, float* vel, float* forces);

	private:
/**************************************************************************************/
		float   magnitudeMax;
/**************************************************************************************/
};

/**************************************************************************************/
/**************************************************************************************/
#endif
