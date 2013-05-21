#ifndef __GAUSSIAN_HEIGHT_FIELD_KERNEL_
#define __GAUSSIAN_HEIGHT_FIELD_KERNEL_

#include <common.cuh>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{
/****************************************************************************************************************************/
	__global__ void GaussianHeightField_calculateHeight_Kernel(double3* m_pos, double A, double x0, double z0, 
								   double p1, double p2, uint nbPos);
/****************************************************************************************************************************/
	__global__ void GaussianHeightField_calculateHeight_Normales_Kernel(double3* m_pos, double3* nx0, double3* nx1,
								   double3* nz0, double3* nz1, double A, double x0, double z0, 
								   double p1, double p2, uint nbPos);
/****************************************************************************************************************************/
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
#endif
/****************************************************************************************************************************/
/****************************************************************************************************************************/