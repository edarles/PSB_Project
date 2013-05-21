#ifndef _MARCHING_CUBE_CUH_
#define _MARCHING_CUBE_CUH_
/**************************************************************************************************************/
/**************************************************************************************************************/
/**************************************************************************************************************/
/**************************************************************************************************************/
#include <Vector3.h>

extern "C"
{
	const uint nbVertexMax = 100000;
	const uint nbVertexMax_perCell = 200;
	const uint nbIndexMax_perCell = 200;

/**************************************************************************************************************/
	void createGrid_MarchingCube_CUDA(Vector3 C, double l, double w, double d, float sizeCell, uint nbPosX, uint nbPosY, uint nbPosZ,
				          double* grid, int* m_nbIndex);
/**************************************************************************************************************/
	void polygonize_MarchingCube_CUDA(double* grid, uint nbPosX, uint nbPosY, uint nbPosZ, double isoLevel, 
				          int* m_nbIndex, double* m_vertex, double* m_normales, int* m_index);
/**************************************************************************************************************/
}
/**************************************************************************************************************/
/**************************************************************************************************************/
#endif
/**************************************************************************************************************/
/**************************************************************************************************************/
