#include <HeuristicMergeSplitt_MixingFluids.h>
/**************************************************************************************/
/**************************************************************************************/
HeuristicMergeSplitt_MixingFluids::HeuristicMergeSplitt_MixingFluids(unsigned int nbBodies)
				  :HeuristicMergeSplitt(nbBodies)
{
	restDensityMax = 0;
}
/**************************************************************************************/
HeuristicMergeSplitt_MixingFluids::HeuristicMergeSplitt_MixingFluids(unsigned int nbBodies, float restDensityMax)
		     		  :HeuristicMergeSplitt(nbBodies)
{
	this->restDensityMax = restDensityMax;
}
/**************************************************************************************/
HeuristicMergeSplitt_MixingFluids::HeuristicMergeSplitt_MixingFluids(const HeuristicMergeSplitt_MixingFluids& H):
				  :HeuristicMergeSplitt(H)
{
	this->restDensityMax = H.restDensityMax;
}
/**************************************************************************************/ 
HeuristicMergeSplitt_MixingFluids::~HeuristicMergeSplitt_MixingFluids()
{
}
/**************************************************************************************/
/**************************************************************************************/
float   HeuristicMergeSplitt_MixingFluids::getRestDensityMax()
{
	return restDensityMax;
}
/**************************************************************************************/
/**************************************************************************************/
void   HeuristicMergeSplitt_MixingFluids::setRestDensityMax(float restDensityMax)
{
	this->restDensityMax = restDensityMax;
}
/**************************************************************************************/
/**************************************************************************************/
void   HeuristicMergeSplitt_MixingFluids::HevaluateFunction(float* pos, float* vel, float* forces)
{
}
/**************************************************************************************/
/**************************************************************************************/
