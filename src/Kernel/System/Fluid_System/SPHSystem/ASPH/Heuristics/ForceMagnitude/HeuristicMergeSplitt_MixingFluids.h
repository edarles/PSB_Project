#ifndef _HEURISTIC_MERGE_SPLITT_MIXING_FLUIDS_
#define _HEURISTIC_MERGE_SPLITT_MIXING_FLUIDS_

#include <HeuristicMergeSplitt.h>

/**************************************************************************************/
/**************************************************************************************/
class HeuristicMergeSplitt_MixingFluids : public HeuristicMergeSplitt
{
	public:
/**************************************************************************************/
		HeuristicMergeSplitt_MixingFluids(unsigned int nbBodies);
		HeuristicMergeSplitt_MixingFluids(unsigned int nbBodies, float magnitudeMax);
		HeuristicMergeSplitt_MixingFluids(const HeuristicMergeSplitt_MixingFluids&);
		~HeuristicMergeSplitt_MixingFluids();

/**************************************************************************************/
/**************************************************************************************/
		float   getRestDensityMax();
/**************************************************************************************/
/**************************************************************************************/
		void   setRestDensityMax(float);
/**************************************************************************************/
/**************************************************************************************/
		void evaluateFunction(float* pos, float* vel, float* forces);

	private:
/**************************************************************************************/
		float   restDensityMax;
/**************************************************************************************/
};

/**************************************************************************************/
/**************************************************************************************/
#endif
