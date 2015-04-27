#include <ForceExt_Trochoide.cuh>
#include <ForceExt_Trochoide_Kernel.cuh>

extern "C"
{
//**************************************************************************************************************************************
//**************************************************************************************************************************************
void evaluate_forceExt_Trochoide_CUDA (double* positions, double* accumForceBuff, double* mass,
   			  	       float A, float k, float theta, float w, float phi, float t, uint nbBodies)
{
    int numBlocksX, numThreadsX;
    computeGridSize(nbBodies, numBlocksX, numThreadsX); 
    evaluate_forceExt_Trochoide_Kernel<<<numBlocksX,numThreadsX>>>((double3*)positions,(double3*)accumForceBuff,mass,
								   A,k,theta,w,phi,t,nbBodies);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
}
