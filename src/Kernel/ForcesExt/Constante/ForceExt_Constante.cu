
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>	 
#include <stdio.h>

#include <host_defines.h>
#include <stdio.h> 
#include <double3.h>

//**************************************************************************************************************************************
//**************************************************************************************************************************************
__global__ void evaluate_force_constante (uint nbBodies, double3* accumForceBuff, double* mass, float3 direction, float amplitude)
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
