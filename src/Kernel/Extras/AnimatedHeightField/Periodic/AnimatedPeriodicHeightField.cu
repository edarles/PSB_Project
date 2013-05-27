#include <AnimatedPeriodicHeightField.cuh>
#include <AnimatedPeriodicHeightField_Kernel.cuh>


/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{

/****************************************************************************************************************************/
/****************************************************************************************************************************/
void AnimatedPeriodicHeightField_calculateHeight_CUDA(double* m_pos, double* A, double* k, double* theta, double* phi, double *omega, double t, uint nbFunc, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        AnimatedPeriodicHeightField_calculateHeight_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, A, k, theta, phi, omega, t, nbFunc, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void AnimatedPeriodicHeightField_calculateHeight_Normales_CUDA(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1,
		double* A, double* k, double* theta, double* phi, double *omega, double t, uint nbFunc, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
       AnimatedPeriodicHeightField_calculateHeight_Normales_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, 
		(double3*) nx0, (double3*) nx1, (double3*) nz0, (double3*) nz1, A,k,theta,phi,omega,t,nbFunc, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
