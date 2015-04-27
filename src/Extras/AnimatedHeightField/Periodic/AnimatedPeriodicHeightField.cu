#include <AnimatedPeriodicHeightField.cuh>
#include <AnimatedPeriodicHeightField_Kernel.cuh>


/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{

/****************************************************************************************************************************/
/****************************************************************************************************************************/
void AnimatedPeriodicHeightField_calculateHeight_CUDA(double* m_pos, double* A, double* k, double* theta, double* phi, 
						      double *omega, double t, uint nbFunc, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        AnimatedPeriodicHeightField_calculateHeight_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, A, k, theta, phi,
										         omega, t, nbFunc, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void AnimatedPeriodicHeightField_calculateHeight_Normales_Gradient_CUDA(double* m_pos, double* m_N, double* A, double* k, double* theta, 
							       double* phi, double *omega, double t, uint nbFunc, uint nbPos)
{
       int numThreadsX, numBlocksX;
       computeGridSize(nbPos,numBlocksX, numThreadsX);
       AnimatedPeriodicHeightField_calculateHeight_Normales_Gradient_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, (double3*) m_N, 
												A,k,theta,phi,omega,t,nbFunc, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
