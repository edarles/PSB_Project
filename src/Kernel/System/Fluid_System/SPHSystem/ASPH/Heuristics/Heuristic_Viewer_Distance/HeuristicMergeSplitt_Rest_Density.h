#ifndef _HEURISTIC_MERGE_SPLITT_REST_DENSITY_
#define _HEURISTIC_MERGE_SPLITT_REST_DENSITY_

#include <HeuristicMergeSplitt.h>

/**************************************************************************************/
/**************************************************************************************/
class HeuristicMergeSplitt_RestDensity : public HeuristicMergeSplitt
{
	public:
/**************************************************************************************/
		HeuristicMergeSplitt_RestDensity(unsigned int nbBodies);
		HeuristicMergeSplitt_RestDensity(unsigned int nbBodies, float restDensityMax);
		HeuristicMergeSplitt_RestDensity(const HeuristicMergeSplitt_RestDensity&);
		~HeuristicMergeSplitt_RestDensity();

/**************************************************************************************/
/**************************************************************************************/
		float getRestDensityMax();

/**************************************************************************************/
/**************************************************************************************/
		void   setRestDensityMax(float);
/**************************************************************************************/
/**************************************************************************************/
		void   evaluateFunction(float* pos, float* vel, float* forces, float* currentRestDensity);

	private:
/**************************************************************************************/
		float   restDensityMax;
/**************************************************************************************/
};

/**************************************************************************************/
/**************************************************************************************/
#endif
