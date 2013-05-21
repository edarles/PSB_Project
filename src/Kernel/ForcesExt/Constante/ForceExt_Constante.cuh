#ifndef _FORCE_EXT_KERNEL_CONSTANTE_H_
#define _FORCE_EXT_KERNEL_CONSTANTE_H_

__global__ void evaluate_force_constante (uint nbBodies, double3* accumForceBuff, double* mass, float3 direction, float amplitude);

#endif
