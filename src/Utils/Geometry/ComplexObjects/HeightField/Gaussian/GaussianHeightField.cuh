#ifndef __GAUSSIAN_HEIGHT_FIELD_CUDA_
#define __GAUSSIAN_HEIGHT_FIELD_CUDA_
/***********************************************************************************************************************************/
/***********************************************************************************************************************************/
#include <common.cuh>
extern "C"
{
/***********************************************************************************************************************************/
	void GaussianHeightField_calculateHeight_CUDA(double* m_pos, double A, double x0, double z0, double p1, double p2, uint nbPos);
/***********************************************************************************************************************************/
	void GaussianHeightField_calculateHeight_Normales_CUDA(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1,
							       double A, double x0, double z0, double p1, double p2, uint nbPos);
}
/***********************************************************************************************************************************/
/***********************************************************************************************************************************/
#endif
/***********************************************************************************************************************************/
/***********************************************************************************************************************************/
