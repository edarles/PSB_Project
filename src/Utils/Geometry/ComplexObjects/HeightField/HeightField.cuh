#ifndef _HEIGHT_FIELD_CUDA_
#define _HEIGHT_FIELD_CUDA_
/*************************************************************************************************************/
/*************************************************************************************************************/
#include <common.cuh>

extern "C"
{
/*************************************************************************************************************/
	void HeightField_initializePos_CUDA(double xC, double yC, double zC, double l, double d, double sizeCellX, double sizeCellZ,
				            uint nbCellsX, uint nbCellsZ, double* m_pos, uint nbPos);
/*************************************************************************************************************/
	void HeightField_initializeHeight_CUDA(double y, double* m_pos, uint nbPos);
/*************************************************************************************************************/
	void HeightField_initializeHeight_Normales_CUDA(double* pos, double* m_x0, double *m_x1, double* m_z0, double* m_z1, 
							double DX, double Y, double DZ, uint nbPos);
}
/*************************************************************************************************************/
/*************************************************************************************************************/
#endif
/*************************************************************************************************************/
/*************************************************************************************************************/