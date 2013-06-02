#ifndef __ANIMATEDPERIODIC_HEIGHT_FIELD_CUDA_
#define __ANIMATEDPERIODIC_HEIGHT_FIELD_CUDA_
/*****************************************************************************/
/*****************************************************************************/
#include <common.cuh>

extern "C"
{
/*****************************************************************************/
/*****************************************************************************/
	void AnimatedPeriodicHeightField_calculateHeight_CUDA(double* m_pos, double* A, double* k, double* theta, double* phi, 
							      double *omega, double t, uint nbFunc, uint nbPos);

/*****************************************************************************/
	void AnimatedPeriodicHeightField_calculateHeight_Normales_Gradient_CUDA(double* m_pos, double *m_N,
							               		double* A, double* k, double* theta, double* phi, double *omega, 
								       		double t, uint nbFunc, uint nbPos);
}
/*****************************************************************************/
/*****************************************************************************/
#endif
/*****************************************************************************/
/*****************************************************************************/
