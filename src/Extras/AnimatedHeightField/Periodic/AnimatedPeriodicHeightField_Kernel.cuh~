#ifndef __ANIMATEDPERIODIC_HEIGHT_FIELD_KERNEL_
#define __ANIMATEDPERIODIC_HEIGHT_FIELD_KERNEL_

#include <common.cuh>

/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{
	__global__ void AnimatedPeriodicHeightField_calculateHeight_Kernel(double3* m_pos, double* A, double* k, double* theta, 
								   double* phi ,double *omega, double t, uint nbFunc, uint nbPos);

/*****************************************************************************************************/
	__global__ void AnimatedPeriodicHeightField_calculateHeight_Normales_Kernel(double3* m_pos, double3* nx0, double3* nx1, 
			  	double3* nz0, double3* nz1, double* A, double* k, double* theta, 
				double* phi, double *omega, double t, uint nbFunc, uint nbPos);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
