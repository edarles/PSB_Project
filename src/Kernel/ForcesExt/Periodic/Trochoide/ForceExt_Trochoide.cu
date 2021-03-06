#include <math_constants.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>	 
#include <stdio.h>

#include <host_defines.h>
#include <stdio.h> 

//**************************************************************************************************************************************
//**************************************************************************************************************************************
__global__ void evaluate_force_trochoide (uint nbBodies, double3* position, double3* accumForceBuffer, double* mass, 
					  float A, float k, float theta, float w, float phi, float t)
{
    int indexP = blockIdx.x * blockDim.x + threadIdx.x;
    if(indexP < nbBodies){
    	double3 pos = position[indexP];
    	double3 currentForces = accumForceBuffer[indexP];
	double masse = mass[indexP];
    	double val =  A*cos(k* (pos.x*cos(theta) + pos.z*sin(theta))*w*t + phi)*masse;
    	accumForceBuffer[indexP] = make_double3(currentForces.x,currentForces.y+val,currentForces.z);
    }
}
/*****************************************************************************************************/
/*****************************************************************************************************/
