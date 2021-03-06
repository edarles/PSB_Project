#ifndef _FORCE_EXT_KERNEL_TROCHOIDE_H_
#define _FORCE_EXT_KERNEL_TROCHOIDE_H_

__global__ void evaluate_force_trochoide (uint nbBodies, double3* position, double3* accumForceBuff, double* mass,
   			  		  float A, float k, float theta, float w, float phi, float t);

#endif
