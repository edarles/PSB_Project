#ifndef __PERIODIC_HEIGHT_FIELD_KERNEL_
#define __PERIODIC_HEIGHT_FIELD_KERNEL_

#include <common.cuh>

/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{
	__global__ void PeriodicHeightField_calculateHeight_Kernel(double3* m_pos, double* A, double* k, double* theta, 
								   double* phi, uint nbFunc, uint nbPos);

/*****************************************************************************************************/
	__global__ void PeriodicHeightField_calculateHeight_Normales_Kernel(double3* m_pos, double3* nx0, double3* nx1, 
			  	double3* nz0, double3* nz1, double* A, double* k, double* theta, 
				double* phi, uint nbFunc, uint nbPos);
/*****************************************************************************************************/
	__global__ void PeriodicHeightField_calculateHeight_Normales_Gradient_Kernel(double3* m_pos, double3* m_N, double* A, 
										     double* k, double* theta, 
								                     double* phi, uint nbFunc, uint nbPos);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
