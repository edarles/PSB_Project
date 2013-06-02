#include <PeriodicHeightField.cuh>
#include <PeriodicHeightField_Kernel.cuh>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{

/****************************************************************************************************************************/
/****************************************************************************************************************************/
void PeriodicHeightField_calculateHeight_CUDA(double* m_pos, double* A, double* k, double* theta, double* phi, uint nbFunc, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        PeriodicHeightField_calculateHeight_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, A, k, theta, phi, nbFunc, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void PeriodicHeightField_calculateHeight_Normales_CUDA(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1,
		double* A, double* k, double* theta, double* phi, uint nbFunc, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        PeriodicHeightField_calculateHeight_Normales_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, 
		(double3*) nx0, (double3*) nx1, (double3*) nz0, (double3*) nz1, A,k,theta,phi,nbFunc, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void PeriodicHeightField_calculateHeight_Normales_Gradient_CUDA(double* m_pos, double* m_N,
								double* A, double* k, double* theta, double* phi, uint nbFunc, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        PeriodicHeightField_calculateHeight_Normales_Gradient_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, (double3*) m_N,
												   A,k,theta,phi,nbFunc, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
