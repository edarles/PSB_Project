#include <HeightField_Kernel.cuh>
#include <stdio.h>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void HeightField_initializePos_Kernel(double xC, double yC, double zC, double l, double d, double sizeCellX, double sizeCellZ,
						 uint nbCellsX, uint nbCellsZ, double3* m_dPos, uint nbPos)
{
	uint indexX =   __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	uint indexZ =   __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	
	if(indexX<nbCellsX && indexZ<nbCellsZ)
	{
		uint indexCell = indexX + indexZ*nbCellsX;
		if(indexCell<nbPos){
			m_dPos[indexCell].x = xC-(l/2)+indexX*sizeCellX;
			m_dPos[indexCell].y = yC;
			m_dPos[indexCell].z = zC-(d/2)+indexZ*sizeCellZ;
		}
	}
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void HeightField_initializeHeight_Kernel(double yC, double3* pos, uint nbPos)
{
	uint index =   __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if(index < nbPos){
		pos[index].y = yC;
	}		
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void HeightField_initializeHeight_Normales_Kernel(double3* pos,
						      double3* m_x0, double3* m_x1, double3* m_z0, double3* m_z1,
						      double x,double y,double z, uint nbPos)
{
	uint index =   __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if(index < nbPos){
		pos[index].y = y;

		m_x0[index].x = pos[index].x + x;
		m_x0[index].y = 0;
		m_x0[index].z = pos[index].z;

		m_x1[index].x = pos[index].x - x;
		m_x1[index].y = 0;
		m_x1[index].z = pos[index].z;

		m_z0[index].x = pos[index].x;
		m_z0[index].y = 0;
		m_z0[index].z = pos[index].z + z;

		m_z1[index].y = pos[index].x;
		m_z1[index].y = 0;
		m_z1[index].z = pos[index].z - z;
		
	}		
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
