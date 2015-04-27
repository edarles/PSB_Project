#ifndef __LINEAR_HEIGHT_FIELD_KERNEL_
#define __LINEAR_HEIGHT_FIELD_KERNEL_

#include <common.cuh>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{
/*****************************************************************************************************/
	__global__ void   LinearHeightField_calculateHeight_Kernel(double3* m_pos, double a, double b, uint nbPos);

/*****************************************************************************************************/
	__global__ void   LinearHeightField_calculateHeight_Normales_Kernel(double3* m_pos, double3* nx0, double3* nx1, 
			  	double3* nz0, double3* nz1, double a, double b, uint nbPos);
}
#endif
