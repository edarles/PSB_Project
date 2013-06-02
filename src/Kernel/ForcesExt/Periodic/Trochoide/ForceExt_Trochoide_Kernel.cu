#include <ForceExt_Trochoide_Kernel.cuh>

extern "C"
{
//**************************************************************************************************************************************
//**************************************************************************************************************************************
__global__ void evaluate_forceExt_Trochoide_Kernel (double3* position, double3* accumForceBuffer, double* mass, 
					  	    float A, float k, float theta, float w, float phi, float t, uint nbBodies)
{
    uint indexP = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(indexP < nbBodies){
    	double3 pos = position[indexP];
    	double val =  A*cos(k* (pos.x*cos(theta) + pos.z*sin(theta))*w*t + phi)*mass[indexP];
    	accumForceBuffer[indexP].y += val;
    }
}
/*****************************************************************************************************/
/*****************************************************************************************************/
}