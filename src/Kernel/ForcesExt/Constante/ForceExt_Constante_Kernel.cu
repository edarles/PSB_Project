#include <ForceExt_Constante_Kernel.cuh>
#include <stdio.h> 
#include <double3.h>

extern "C"
{
//**************************************************************************************************************************************
//**************************************************************************************************************************************
__global__ void evaluate_force_constante_Kernel(double3* accumForceBuff, double* mass, float3 direction, float amplitude, uint nbBodies)
{
    uint indexP = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(indexP<nbBodies){
	    accumForceBuff[indexP].x += (direction.x*amplitude)*mass[indexP];
	    accumForceBuff[indexP].y += (direction.y*amplitude)*mass[indexP];
	    accumForceBuff[indexP].z += (direction.z*amplitude)*mass[indexP];
    }
}
/*****************************************************************************************************/
/*****************************************************************************************************/
}
