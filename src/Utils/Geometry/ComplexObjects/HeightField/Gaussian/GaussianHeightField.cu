#include <GaussianHeightField.cuh>
#include <GaussianHeightField_Kernel.cuh>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{

/****************************************************************************************************************************/
/****************************************************************************************************************************/
void GaussianHeightField_calculateHeight_CUDA(double* m_pos, double A, double x0, double z0, double p1, double p2, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        GaussianHeightField_calculateHeight_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, A, x0, z0, p1, p2, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void GaussianHeightField_calculateHeight_Normales_CUDA(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1,
						       double A, double x0, double z0, double p1, double p2, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        GaussianHeightField_calculateHeight_Normales_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, 
			(double3*) nx0, (double3*) nx1, (double3*) nz0, (double3*) nz1, A, x0, z0, p1, p2, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
