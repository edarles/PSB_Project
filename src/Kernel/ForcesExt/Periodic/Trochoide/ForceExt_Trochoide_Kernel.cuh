#ifndef _FORCE_EXT_TROCHOIDE_KERNEL_H_
#define _FORCE_EXT_TROCHOIDE_KERNEL_H_
/******************************************************************************************/
/******************************************************************************************/

#include <cuda.h>
#include <vector_functions.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <host_defines.h>

extern "C"
{
/******************************************************************************************/
__global__ void evaluate_forceExt_Trochoide_Kernel (double3* position, double3* accumForceBuff, double* mass,
   			  		  	    float A, float k, float theta, float w, float phi, float t, uint nbBodies);
/******************************************************************************************/
}
/******************************************************************************************/
/******************************************************************************************/
#endif
/******************************************************************************************/
/******************************************************************************************/
