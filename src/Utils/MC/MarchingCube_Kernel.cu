#include <MarchingCube_Kernel.cuh>
#include <MarchingCube.cuh>
#include <UniformGrid.cuh>
#include <AtomicDoubleAdd.cuh>
#include <UniformGridKernel.cuh>

#include <stdio.h>
/**************************************************************************************************************/
/**************************************************************************************************************/
extern "C"
{
/**************************************************************************************************************/
/**************************************************************************************************************/
__global__ void createGrid_MarchingCube_Kernel(double4* grid, double3 center, double l, double w, double d,
					       float sizeCell, uint nbPosX, uint nbPosY, uint nbPosZ, int* m_nbIndex)
{
	uint indexX =   __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	uint indexY =   __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	uint indexZ =   __mul24(blockIdx.z,blockDim.z) + threadIdx.z;

	if(indexX<nbPosX && indexY<nbPosY && indexZ<nbPosZ)
	{
		uint indexCell = indexX + indexY*nbPosX + indexZ*nbPosX*nbPosY;
		double3 pos = make_double3(center.x-(l/2)+(sizeCell/2)+indexX*sizeCell,
			      center.y-(w/2)+(sizeCell/2)+indexY*sizeCell,center.z-(d/2)+(sizeCell/2)+indexZ*sizeCell);
		grid[indexCell].x = pos.x;
		grid[indexCell].y = pos.y;
		grid[indexCell].z = pos.z;
		grid[indexCell].w = 0;
		m_nbIndex[indexCell] = 0;
		//printf("pos:%f %f %f\n",grid[indexCell].x,grid[indexCell].y,grid[indexCell].z);
	}
}
/**************************************************************************************************************/
/**************************************************************************************************************/
__global__ void polygonize_MarchingCube_Kernel(double4* grid, uint nbCellsX, uint nbCellsY, uint nbCellsZ,
					       double isoLevel, int* nbIndex, double3* vertexs, double3* normales, int* indexs)
{
	int indexX =   __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int indexY =   __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
	int indexZ =   __mul24(blockIdx.z,blockDim.z) + threadIdx.z;

	if(indexX>=0 && indexY>=0 && indexZ>=0 && indexX<(nbCellsX-1) && indexY<(nbCellsY-1) && indexZ<(nbCellsZ-1))
	{
		uint i0 = indexX + indexY*nbCellsX + indexZ*nbCellsX*nbCellsY;
		uint i3 = indexX + indexY*nbCellsX + (indexZ+1)*nbCellsX*nbCellsY;
		uint i4 = indexX + (indexY+1)*nbCellsX + indexZ*nbCellsX*nbCellsY;
		uint i7 = indexX + (indexY+1)*nbCellsX + (indexZ+1)*nbCellsX*nbCellsY;

		uint i1 = (indexX+1) + indexY*nbCellsX + indexZ*nbCellsX*nbCellsY;
		uint i2 = (indexX+1) + indexY*nbCellsX + (indexZ+1)*nbCellsX*nbCellsY;
		uint i5 = (indexX+1) + (indexY+1)*nbCellsX + indexZ*nbCellsX*nbCellsY;
		uint i6 = (indexX+1) + (indexY+1)*nbCellsX + (indexZ+1)*nbCellsX*nbCellsY;
		
		double3 P0 = make_double3(grid[i0].x,grid[i0].y,grid[i0].z);
		double3 P1 = make_double3(grid[i1].x,grid[i1].y,grid[i1].z);
		double3 P2 = make_double3(grid[i2].x,grid[i2].y,grid[i2].z);
		double3 P3 = make_double3(grid[i3].x,grid[i3].y,grid[i3].z);
		double3 P4 = make_double3(grid[i4].x,grid[i4].y,grid[i4].z);
		double3 P5 = make_double3(grid[i5].x,grid[i5].y,grid[i5].z);
		double3 P6 = make_double3(grid[i6].x,grid[i6].y,grid[i6].z);
		double3 P7 = make_double3(grid[i7].x,grid[i7].y,grid[i7].z);

		double val0 = grid[i0].w;
		double val1 = grid[i1].w;
		double val2 = grid[i2].w;
		double val3 = grid[i3].w;
		double val4 = grid[i4].w;
		double val5 = grid[i5].w;
		double val6 = grid[i6].w;
		double val7 = grid[i7].w; 

		uint CubeIndex = 0;
	        if (grid[i0].w <= isoLevel) CubeIndex |= 1;
    	        if (grid[i1].w <= isoLevel) CubeIndex |= 2;
		if (grid[i2].w <= isoLevel) CubeIndex |= 4;
    	        if (grid[i3].w <= isoLevel) CubeIndex |= 8;
		if (grid[i4].w <= isoLevel) CubeIndex |= 16;
    	        if (grid[i5].w <= isoLevel) CubeIndex |= 32;
		if (grid[i6].w <= isoLevel) CubeIndex |= 64;
    	        if (grid[i7].w <= isoLevel) CubeIndex |= 128;
    
		//if(CubeIndex!=0)printf("CubeIndex:%d\n",CubeIndex);
		uint size = nbCellsX + nbCellsY*nbCellsX + nbCellsZ*nbCellsX*nbCellsY;
		uint indexCell = indexX + indexY*nbCellsX + indexZ*nbCellsX*nbCellsY;
		if (EdgeTable[CubeIndex] != 0){
			 polygoniseCell_MarchingCube(P0,P1,P2,P3,P4,P5,P6,P7,
					val0, val1, val2, val3, val4, val5, val6, val7,
					CubeIndex, isoLevel, indexCell, size, nbIndex, vertexs, normales, indexs);
			
		}
		//if(nbIndex[indexCell]!=0) printf("nbIndex:%d\n",nbIndex[indexCell]);
	}
}
/**************************************************************************************************************/
/**************************************************************************************************************/
__device__ void polygoniseCell_MarchingCube(double3 pG0, double3 pG1, double3 pG2, double3 pG3, double3 pG4,
					double3 pG5, double3 pG6, double3 pG7, double val0, double val1, double val2,
					double val3, double val4, double val5, double val6, double val7,
					int CubeIndex, double isoLevel, uint indexCell, uint size, int* nbIndex, double3* vertexs, 
					double3* normales, int* indexs)
{
	double3 P[12];
	if (EdgeTable[CubeIndex] & 1)
		P[0] = vertexInterpolate_MarchingCube(pG0,pG1,val0,val1,isoLevel);
	if (EdgeTable[CubeIndex] & 2)
		P[1] = vertexInterpolate_MarchingCube(pG1,pG2,val1,val2,isoLevel);
	if (EdgeTable[CubeIndex] & 4)
		P[2] = vertexInterpolate_MarchingCube(pG2,pG3,val2,val3,isoLevel);
	if (EdgeTable[CubeIndex] & 8)
		P[3] = vertexInterpolate_MarchingCube(pG3,pG0,val3,val0,isoLevel);
	if (EdgeTable[CubeIndex] & 16)
		P[4] = vertexInterpolate_MarchingCube(pG4,pG5,val4,val5,isoLevel);
	if (EdgeTable[CubeIndex] & 32)
		P[5] = vertexInterpolate_MarchingCube(pG5,pG6,val5,val6,isoLevel);
	if (EdgeTable[CubeIndex] & 64)
		P[6] = vertexInterpolate_MarchingCube(pG6,pG7,val6,val7,isoLevel);
	if (EdgeTable[CubeIndex] & 128)
		P[7] = vertexInterpolate_MarchingCube(pG7,pG4,val7,val4,isoLevel);
	if (EdgeTable[CubeIndex] & 256)
		P[8] = vertexInterpolate_MarchingCube(pG0,pG4,val0,val4,isoLevel);
	if (EdgeTable[CubeIndex] & 512)
		P[9] = vertexInterpolate_MarchingCube(pG1,pG5,val1,val5,isoLevel);
	if (EdgeTable[CubeIndex] & 1024)
		P[10] = vertexInterpolate_MarchingCube(pG2,pG6,val2,val6,isoLevel);
	if (EdgeTable[CubeIndex] & 2048)
		P[11] = vertexInterpolate_MarchingCube(pG3,pG7,val3,val7,isoLevel);

	uint nbT = 0;
	for (int i=0;TriTable[CubeIndex][i]!=-1;i+=3)
	{
		for(int j=0;j<3;j++)
		{
			uint nbInd = nbIndex[indexCell];
			nbIndex[indexCell] = nbIndex[indexCell] + 1; 
			indexs[indexCell*nbIndexMax_perCell + nbInd] = nbInd;
			vertexs[indexCell*nbVertexMax_perCell + nbInd] = P[TriTable[CubeIndex][i+j]];
		}
		nbT++;
	}
}
/**************************************************************************************************************/
/**************************************************************************************************************/
__device__ double3 vertexInterpolate_MarchingCube(double3 P1, double3 P2, double ValP1,double ValP2, double isoLevel)
{
   if (fabs(isoLevel-ValP1) < 0.000001)	return(P1); 
   if (fabs(isoLevel-ValP2) < 0.000001)	return(P2); 
   if (fabs(ValP1-ValP2) < 0.000001)	return(P1);
   double mu = (isoLevel - ValP1) / (ValP2 - ValP1);
   double3 P;
   P.x = P1.x + mu * (P2.x - P1.x);
   P.y = P1.y + mu * (P2.y - P1.y);
   P.z = P1.z + mu * (P2.z - P1.z);
   return(P);
}
/**************************************************************************************************************/
/**************************************************************************************************************/
}
