#include <UniformGrid.cuh>
#include <UniformGridKernel.cuh>
#include <cuda.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdint.h>
#include <QDataStream>
#include <QFile>

/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{

/****************************************************************************************************************************/
/****************************************************************************************************************************/	
void createGrid(double xC, double yC, double zC, double length, double width, double depth, float sizeCell, UniformGrid *grid)
{
   grid->Min.x = xC-length/2;
   grid->Min.y = yC-width/2;
   grid->Min.z = zC-depth/2;

   grid->Max.x = xC+length/2;
   grid->Max.y = yC+width/2;
   grid->Max.z = zC+depth/2;

   grid->nbCellsX = ceil(length/sizeCell);
   grid->nbCellsY = ceil(width/sizeCell);
   grid->nbCellsZ = ceil(depth/sizeCell);

   grid->sizeCell = sizeCell;
   
   allocateArray((void**)&grid->m_dPos, sizeof(double)*3*(grid->nbCellsX + grid->nbCellsX*grid->nbCellsY + grid->nbCellsZ*grid->nbCellsY*grid->nbCellsX)); 
   allocateArray((void**)&grid->indexParticles, sizeof(int)*MAXPARTICLES*(grid->nbCellsX + grid->nbCellsX*grid->nbCellsY + grid->nbCellsZ*grid->nbCellsY*grid->nbCellsX));
   allocateArray((void**)&grid->nbParticles, sizeof(int)*(grid->nbCellsX + grid->nbCellsX*grid->nbCellsY + grid->nbCellsZ*grid->nbCellsY*grid->nbCellsX));
   
   int numThreadsX, numBlocksX, numThreadsY, numBlocksY,numThreadsZ, numBlocksZ;
   computeGridSize(grid->nbCellsX,numBlocksX, numThreadsX);
   computeGridSize(grid->nbCellsY,numBlocksY, numThreadsY);
   computeGridSize(grid->nbCellsZ,numBlocksZ, numThreadsZ);

   dim3 dimBlock(numBlocksX,numBlocksY,numBlocksZ);
   dim3 dimThreads(numThreadsX,numThreadsY,numThreadsZ);
 
   createCell_Kernel<<<dimThreads,dimBlock>>>(xC,yC,zC,length, width, depth, sizeCell, 
				              grid->nbCellsX, grid->nbCellsY, grid->nbCellsZ, (double3*)grid->m_dPos,
					      grid->nbParticles);
   printf("nb:%d %d %d\n",grid->nbCellsX,grid->nbCellsY,grid->nbCellsZ);
}

/****************************************************************************************************************************/
/****************************************************************************************************************************/
void _finalizeGrid(UniformGrid *grid)
{
	delete[] grid->m_hPos;
	delete[] grid->m_hIsoVal;
	freeArray(grid->m_dPos);
	freeArray(grid->nbParticles);
	freeArray(grid->indexParticles);
	freeArray(grid->m_dIsoVal);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void reinitCells(UniformGrid* grid)
{
   int numThreadsX, numBlocksX, numThreadsY, numBlocksY,numThreadsZ, numBlocksZ;
   computeGridSize(grid->nbCellsX,numBlocksX, numThreadsX);
   computeGridSize(grid->nbCellsY,numBlocksY, numThreadsY);
   computeGridSize(grid->nbCellsZ,numBlocksZ, numThreadsZ);

   dim3 dimBlock(numBlocksX,numBlocksY,numBlocksZ);
   dim3 dimThreads(numThreadsX,numThreadsY,numThreadsZ);

  // grid->nbCellsX = grid->nbCellsZ = grid->nbCellsY;
   reinitCells_Kernel<<<dimThreads,dimBlock>>>(grid->nbParticles,grid->nbCellsX,grid->nbCellsY,grid->nbCellsZ);
  
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void storeParticles(UniformGrid* grid, double *pos, unsigned int nbBodies)
{
   int numThreadsX, numBlocksX;
   computeGridSize(nbBodies,numBlocksX, numThreadsX);
   storeParticles_Kernel<<<numBlocksX,numThreadsX>>>((double3*)pos,nbBodies,grid->nbParticles,grid->indexParticles,grid->sizeCell,
		                                   grid->Min, grid->nbCellsX,grid->nbCellsY,grid->nbCellsZ);
    //CUT_CHECK_ERROR("Store particles Kernel execution failed");*/
}

/****************************************************************************************************************************/
/****************************************************************************************************************************/
void searchNeighbooring(double* pos, double* radius, UniformGrid *grid, double scale, partVoisine voisines, unsigned int nbBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    searchNeighbooring_Kernel<<<numBlocksX,numThreadsX>>>((double3*) pos, radius, grid->indexParticles,grid->nbParticles,grid->nbCellsX, 
						       grid->nbCellsY, grid->nbCellsZ, grid->sizeCell, grid->Min, scale,
						       voisines,nbBodies);
    //CUT_CHECK_ERROR("Search neighbooring Kernel execution failed");*/
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void evaluateIso_CUDA(double* posMC, uint nbCellsX_MC, uint nbCellsY_MC, uint nbCellsZ_MC, double scale,
	         UniformGrid *grid, double* pos, double* radius, uint nbBodies)
{
   int numThreadsX, numBlocksX, numThreadsY, numBlocksY,numThreadsZ, numBlocksZ;
   computeGridSize(nbCellsX_MC,numBlocksX, numThreadsX);
   computeGridSize(nbCellsY_MC,numBlocksY, numThreadsY);
   computeGridSize(nbCellsZ_MC,numBlocksZ, numThreadsZ);

   dim3 dimBlock(numBlocksX,numBlocksY,numBlocksZ);
   dim3 dimThreads(numThreadsX,numThreadsY,numThreadsZ);

   evaluateIso_Kernel<<<dimThreads,dimBlock>>>((double4*) posMC, nbCellsX_MC, nbCellsY_MC, nbCellsZ_MC, scale,
			           grid->Min, grid->sizeCell, grid->nbCellsX, grid->nbCellsY, grid->nbCellsZ, 
			           grid->nbParticles, grid->indexParticles,
				   (double3*) pos, radius, nbBodies);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void computeNormales_Vertex_CUDA(double* posV, double* normales, float scale, uint nbV,
			         UniformGrid *grid, double* pos, double* radius, double* mass, double* densities, uint nbBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbV,numBlocksX, numThreadsX);

    computeNormales_Vertexs_Kernel<<<numBlocksX,numThreadsX>>>((double3*) posV, (double3*) normales, scale, nbV,
			                       		  grid->Min, grid->sizeCell, grid->nbCellsX, grid->nbCellsY, grid->nbCellsZ, 
					       		  grid->nbParticles, grid->indexParticles,
				               		  (double3*) pos, radius, mass, densities, nbBodies);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void displayGrid(UniformGrid *grid)
{
	glColor3f(1,1,1);
	float s = grid->sizeCell;
        grid->nbCellsZ = grid->nbCellsY;
       // printf("size:%f nbC:%d %d %d\n",s,grid.nbCellsX,grid.nbCellsY,grid.nbCellsZ);
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	
	for(unsigned int i=0;i<grid->nbCellsX;i++){
		for(unsigned int j=0;j<grid->nbCellsY;j++){
			for(unsigned int k=0;k<grid->nbCellsZ;k++){
				uint indexCell = i + j*grid->nbCellsX + k*grid->nbCellsX*grid->nbCellsY;
				double xC = grid->m_hPos[indexCell*3];
				double yC = grid->m_hPos[indexCell*3+1];
				double zC = grid->m_hPos[indexCell*3+2];
	//			printf("index:%d xC:%f yC:%f zC:%f\n",indexCell,xC,yC,zC);

				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC+s/2,zC+s/2);
				glVertex3f(xC-s/2,yC+s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC-s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC+s/2);
				glVertex3f(xC-s/2,yC-s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC+s/2,yC+s/2,zC+s/2);
				glVertex3f(xC+s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC+s/2,zC+s/2);
				glVertex3f(xC-s/2,yC+s/2,zC-s/2);
				glVertex3f(xC-s/2,yC-s/2,zC-s/2);
				glVertex3f(xC-s/2,yC-s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC+s/2,zC+s/2);
				glVertex3f(xC+s/2,yC+s/2,zC+s/2);
				glVertex3f(xC+s/2,yC-s/2,zC+s/2);
				glVertex3f(xC-s/2,yC-s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC-s/2);
				glVertex3f(xC-s/2,yC-s/2,zC-s/2);
				glEnd();
			}
		}
	}
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void displayGridByIso(UniformGrid *grid)
{
	glColor3f(1,1,1);
	float s = grid->sizeCell;
        grid->nbCellsZ = grid->nbCellsY;
       // printf("size:%f nbC:%d %d %d\n",s,grid.nbCellsX,grid.nbCellsY,grid.nbCellsZ);
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINES);
	
	for(unsigned int i=0;i<grid->nbCellsX;i++){
		for(unsigned int j=0;j<grid->nbCellsY;j++){
			for(unsigned int k=0;k<grid->nbCellsZ;k++){
				uint indexCell = i + j*grid->nbCellsX + k*grid->nbCellsX*grid->nbCellsY;
				float val = grid->m_hIsoVal[indexCell];//1-grid->m_hIsoVal[indexCell];
				
				if(val!=1000 && val<0.0001){// && val<0.1){
					//printf("val:%f\n",val);
				val = 1 - val;
				double xC = grid->m_hPos[indexCell*3];
				double yC = grid->m_hPos[indexCell*3+1];
				double zC = grid->m_hPos[indexCell*3+2];

				double nx = grid->m_hIsoNorm[indexCell*3];
				double ny = grid->m_hIsoNorm[indexCell*3+1];
				double nz = grid->m_hIsoNorm[indexCell*3+2];

				glColor3f(0,1,0);
				glBegin(GL_LINES);
				glVertex3f(xC,yC,zC);
				glVertex3f(xC+nx*0.03,yC+ny*0.03,zC+nz*0.03);
				glEnd();
				
				glColor3f(val,0,0);
	//			printf("index:%d xC:%f yC:%f zC:%f\n",indexCell,xC,yC,zC);

				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC+s/2,zC+s/2);
				glVertex3f(xC-s/2,yC+s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC-s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC+s/2);
				glVertex3f(xC-s/2,yC-s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC+s/2,yC+s/2,zC+s/2);
				glVertex3f(xC+s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC+s/2,zC+s/2);
				glVertex3f(xC-s/2,yC+s/2,zC-s/2);
				glVertex3f(xC-s/2,yC-s/2,zC-s/2);
				glVertex3f(xC-s/2,yC-s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC+s/2,zC+s/2);
				glVertex3f(xC+s/2,yC+s/2,zC+s/2);
				glVertex3f(xC+s/2,yC-s/2,zC+s/2);
				glVertex3f(xC-s/2,yC-s/2,zC+s/2);
				glEnd();
				glBegin(GL_POLYGON);
				glVertex3f(xC-s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC+s/2,zC-s/2);
				glVertex3f(xC+s/2,yC-s/2,zC-s/2);
				glVertex3f(xC-s/2,yC-s/2,zC-s/2);
				glEnd();
				}
			}
		}
	}
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void _exportData(const char* filename, UniformGrid *grid)
{   
	FILE *f = fopen(filename,"wb");
	char header[3];
	header[0] = 'V'; header[1] = 'O'; header[2] = 'L';
	for(unsigned int i=0;i<3;i++)
		fwrite(&header[i],sizeof(char),1,f);
	uint8_t version = 3;
	uint32_t encoding = 1;
	uint32_t nbChannels = 1;
	
	fwrite(&version,sizeof(uint8_t),1,f);
	fwrite(&encoding,sizeof(uint32_t),1,f);
	fwrite(&grid->nbCellsZ,sizeof(uint32_t),1,f);
	fwrite(&grid->nbCellsZ,sizeof(uint32_t),1,f);
	fwrite(&grid->nbCellsZ,sizeof(uint32_t),1,f);
	fwrite(&nbChannels,sizeof(uint32_t),1,f);

	fwrite(&grid->Min.x,sizeof(float),1,f); fwrite(&grid->Min.y,sizeof(float),1,f); fwrite(&grid->Min.z,sizeof(float),1,f);
	fwrite(&grid->Max.x,sizeof(float),1,f); fwrite(&grid->Max.y,sizeof(float),1,f); fwrite(&grid->Max.z,sizeof(float),1,f);

	for(int i=0;i<grid->nbCellsZ;i++){
		for(int j=0;j<grid->nbCellsZ;j++){
			for(int k=0;k<grid->nbCellsZ;k++){
				int indexCell = i + j*grid->nbCellsZ + k*grid->nbCellsZ*grid->nbCellsZ;
				float val = grid->m_hIsoVal[indexCell];
				fwrite(&val,sizeof(float),1,f);
			}
		}
	}
	fclose(f);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
