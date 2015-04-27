#include <HeuristicsMergeSplitt.h>

/**************************************************************************************/
/**************************************************************************************/
HeuristicsMergeSplitt::HeuristicsMergeSplitt()
{
	heuristics.clear();
}
/**************************************************************************************/
HeuristicsMergeSplitt::HeuristicsMergeSplitt(vector<HeuristicsMergeSplitt> heuristics)
{
	this->heuristics.clear();
	for(unsigned int i=0;i<heuristics.size();i++)
		this->heuristics.push_back(heuristics[i]);
}
/**************************************************************************************/
HeuristicsMergeSplitt::~HeristicsMergeSplitt()
{
	for(unsigned int i=0;i<heuristics.size();i++)
		delete(heuristics[i];
	heuristics.clear();
}
/**************************************************************************************/
/**************************************************************************************/
HeuristicMergeSplitt* HeuristicsMergeSplitt::getHeuristic(unsigned int i)
{
	assert(i<heuristics.size());
	return heuristics[i];
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicsMergeSplitt::addHeuristic(HeuristicsMergeSplitt* H)
{
	heuristics.push_back(H);
}
/**************************************************************************************/
void HeuristicsMergeSplitt::removeHeuristic()
{
	heuristics.pop_back();
}
/**************************************************************************************/
void HeuristicsMergeSplitt::removeHeuristic(unsigned int i)
{
	assert(i<heuristics.size());
	vector<HeuristicMergeSplitt*> newHeuristics;
	for(unsigned int j=0;j<heuristics.size();j++){
		if(i!=j)
			newHeuristics.push_back(heuristics[j]);
	}
	heuristics.clear();
	for(unsigned int j=0;j<newHeuristics.size();j++)
		this->heuristics.push_back(newHeuristics[j]);
}
/**************************************************************************************/
/**************************************************************************************/
bool* HeuristicsMergeSplitt::evaluationFunction(float* pos, float* vel, float *forces)
{
	if(heuristics.size()>0){
		bool *res = new bool[heuristics[i]->getNbBodies()];
		for(unsigned int i=0;i<heuristics.size();i++){
			heuristics[i]->evaluateFunction(pos,vel,forces,res);
			if(i==0){
				bool *res0 = heuristics[0]->getResult();
				for(unsigned int j=0;j<heuristics[0]->getNbBodies();j++)
					res[j] = res0[j]; 
			}
			else{
				bool *resI = heuristics[i]->getResult();
				for(unsigned int j=0;j<heuristics[i]->getNbBodies();j++)
					res[j] = res[j] && resI[j];
			}
		}
		return res;
	}
	return NULL;
}
/**************************************************************************************/
/**************************************************************************************/
