#ifndef __PERIODIC_HEIGHT_FIELD_CUDA_
#define __PERIODIC_HEIGHT_FIELD_CUDA_
/*****************************************************************************/
/*****************************************************************************/
#include <common.cuh>

extern "C"
{
/*****************************************************************************/
/*****************************************************************************/
	void PeriodicHeightField_calculateHeight_CUDA(double* m_pos, double* A, double* k, double* theta, double* phi, uint nbFunc, uint nbPos);

/*****************************************************************************/
	void PeriodicHeightField_calculateHeight_Normales_CUDA(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1,
							       double* A, double* k, double* theta, double* phi, uint nbFunc, uint nbPos);
}
/*****************************************************************************/
/*****************************************************************************/
#endif
/*****************************************************************************/
/*****************************************************************************/
