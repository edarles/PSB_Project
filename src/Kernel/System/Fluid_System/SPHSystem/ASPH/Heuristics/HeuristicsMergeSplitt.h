#ifndef _HEURISTICS_MERGE_SPLITT_
#define _HEURISTICS_MERGE_SPLITT_

#include <HeuristicMergeSplitt.h>
#include <vector.h>
using namespace std;

/**************************************************************************************/
/**************************************************************************************/
class HeuristicsMergeSplitt 
{
	public:
	/**************************************************************************************/
		HeuristicsMergeSplitt();
		HeuristicsMergeSplitt(vector<HeuristicsMergeSplitt>);
		~HeristicsMergeSplitt();
	/**************************************************************************************/
	/**************************************************************************************/
		HeuristicMergeSplitt* getHeuristic(unsigned int);
	/**************************************************************************************/
	/**************************************************************************************/
		void addHeuristic(HeuristicsMergeSplitt*);
		void removeHeuristic();
		void removeHeuristic(unsigned int);
	/**************************************************************************************/
	/**************************************************************************************/
		bool* evaluationFunction(float* pos, float* vel, float *forces);
	/**************************************************************************************/
	/**************************************************************************************/
	protected:
		vector<HeuristicMergeSplitt*> heuristics;
};
/**************************************************************************************/
/**************************************************************************************/
#endif
