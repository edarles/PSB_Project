#include <HeightField.cuh>
#include <HeightField_Kernel.cuh>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void HeightField_initializePos_CUDA(double xC, double yC, double zC, double l, double d, double sizeCellX, double sizeCellZ,
				    uint nbCellsX, uint nbCellsZ, double* m_pos, uint nbPos)
{
	int numThreadsX, numThreadsZ, numBlocksX, numBlocksZ;
   	computeGridSize(nbCellsX,numBlocksX, numThreadsX);
	computeGridSize(nbCellsZ,numBlocksZ, numThreadsZ);

	dim3 dimBlock(numBlocksX,numBlocksZ);
   	dim3 dimThreads(numThreadsX,numThreadsZ);

        HeightField_initializePos_Kernel<<<dimThreads,dimBlock>>>(xC,yC,zC,l,d,sizeCellX,sizeCellZ,nbCellsX,nbCellsZ,(double3*) m_pos, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void HeightField_initializeHeight_CUDA(double y, double* m_pos, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX, numThreadsX);
        HeightField_initializeHeight_Kernel<<<numBlocksX,numThreadsX>>>(y, (double3*) m_pos, nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void HeightField_initializeHeight_Normales_CUDA(double* pos, double* m_x0, double *m_x1, double* m_z0, double* m_z1, 
						double DX, double Y, double DZ, uint nbPos)
{
	int numThreadsX, numBlocksX;
   	computeGridSize(nbPos,numBlocksX,numThreadsX);
        HeightField_initializeHeight_Normales_Kernel<<<numBlocksX,numThreadsX>>>
		((double3*)pos, (double3*) m_x0, (double3*) m_x1, (double3*) m_z0, (double3*) m_z1, DX,Y,DZ,nbPos);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
