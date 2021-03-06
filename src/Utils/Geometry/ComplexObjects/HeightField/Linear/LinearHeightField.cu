#include <LinearHeightField.cuh>
#include <LinearHeightField_Kernel.cuh>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{

/****************************************************************************************************************************/
/****************************************************************************************************************************/
void LinearHeightField_calculateHeight_CUDA(double* m_pos, double a, double b, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        LinearHeightField_calculateHeight_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, a, b, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void LinearHeightField_calculateHeight_Normales_CUDA(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1,
		double a, double b, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        LinearHeightField_calculateHeight_Normales_Kernel<<<numBlocksX, numThreadsX>>>((double3*) m_pos, 
		(double3*) nx0, (double3*) nx1, (double3*) nz0, (double3*) nz1, a, b, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
