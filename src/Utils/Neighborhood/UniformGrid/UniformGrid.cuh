#ifndef __UNIFORM_GRID_CUDA_
#define __UNIFORM_GRID_CUDA_
/*****************************************************************************/
/*****************************************************************************/
#include <common.cuh>
#include <SphKernel.cuh>

#define MAXPARTICLES 64

extern "C"
{
#pragma once

/*****************************************************************************/
/*****************************************************************************/
typedef struct
{
	float3 Min;
	float3 Max;
	float sizeCell;
	double l,h,d;
	int* indexParticles;
	int* nbParticles;
	double3* m_dPos; 
        double* m_hPos;
        float* m_dIsoVal;
        float* m_hIsoVal;  
        float* m_dIsoNorm;
        float* m_hIsoNorm;     
	int nbCellsX;
	int nbCellsY;
	int nbCellsZ;
}  UniformGrid;
/*****************************************************************************/
/*** INITIALIZE UNIFORM GRID *************************************************/
/*****************************************************************************/

void createGrid(double xC, double yC, double zC, double length, double width, double depth, float sizeCell, UniformGrid *grid);

void reinitCells(UniformGrid* grid);

/*****************************************************************************/
void _finalizeGrid(UniformGrid *grid);

/*****************************************************************************/
/*** STORE PARTICLES POSITIONS ***********************************************/
/*****************************************************************************/
void storeParticles(UniformGrid* grid, double *pos, unsigned int nbBodies);

/*****************************************************************************/
/*** CALCULATE NEIGHBOORING PARTICLES ****************************************/
/*****************************************************************************/
void searchNeighbooring(double* pos, double* radius, UniformGrid *grid, double scale, partVoisine voisines, unsigned int nbBodies);

/*****************************************************************************/
/*** EVALUATE ISO DENSITIES********** ****************************************/
/*****************************************************************************/
void evaluateIso_CUDA(double* posMC, uint nbCellsX_MC, uint nbCellsY_MC, uint nbCellsZ_MC, double scale,
	              UniformGrid *grid, double* pos, double* radius, uint nbBodies);

/*****************************************************************************/
/*** COMPUTE NORMALE AT VERTEXS **********************************************/
/*****************************************************************************/
void computeNormales_Vertex_CUDA(double* posV, double* normales, float scale, uint nbV,
			         UniformGrid *grid, double* pos, double* radius, double* mass, double* densities, uint nbBodies);

/*****************************************************************************/
/*** DISPLAY UNIFORM GRID ****************************************/
/*****************************************************************************/
void displayGrid(UniformGrid *grid);
void displayGridByIso(UniformGrid *grid);

/*****************************************************************************/
/*** EXPORT TO RENDER FLUID IN MITSUBA (HETEREGENOUS PARTICIPATING MEDIA *****/
/*****************************************************************************/
void _exportData(const char* filename, UniformGrid *grid, double* positions, double* radius, double* m_densities, uint nbParticles);
void _exportAlbedo(const char* filename, UniformGrid *grid, double* positions, double* radius, double* m_a);
}
/*****************************************************************************/
/*****************************************************************************/
#endif
