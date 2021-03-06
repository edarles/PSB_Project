#include <HeightField.h>
#include <HeightField.cuh>
#include <cstring>
#include <stdio.h>
#include <GL/gl.h>
#include <common.cuh>
/*****************************************************************************************************/
/*****************************************************************************************************/
namespace Utils 
{
/*****************************************************************************************************/
/*****************************************************************************************************/
HeightField::HeightField()
	    :ObjectGeo(),Min(-1,-1,-1), Max(1,1,1), hMin(-1.0), hMax(1.0), dx(0.05), dz(0.05)
{
	create();
}
/*****************************************************************************************************/
HeightField::HeightField(Vector3 min, Vector3 max, double d_x, double d_z)
	      :ObjectGeo(),Min(min), Max(max), dx(d_x), dz(d_z)
{
	create();
}
/*****************************************************************************************************/
HeightField::HeightField(const HeightField& H):ObjectGeo(H)
{
	this->Min = H.Min;
	this->Max = H.Max;
	this->dx = H.dx;
	this->dz = H.dz;
	this->nbPosX = H.nbPosX;
	this->nbPosZ = H.nbPosZ;
	this->center = H.center;
	if(pos!=NULL)
		delete[] pos;
	uint size = nbPosX + nbPosX*nbPosZ;
	pos = new double[3*size];
	memset(pos, 0, 3*size*sizeof(double));
	if(H.pos!=NULL){
		for(uint nb = 0; nb < size; nb++){
			pos[nb*3] = H.pos[nb*3];
			pos[nb*3+1] = H.pos[nb*3+1];
			pos[nb*3+2] = H.pos[nb*3+2];
		}
	}
}
/*****************************************************************************************************/
HeightField::~HeightField()
{
	delete[] pos;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void HeightField::create()
{
	assert(Max.x()>Min.x() && Max.z()>Min.z());
	center = Vector3((Max.x()+Min.x())/2,(Max.y()+Min.y())/2,(Max.z()+Min.z())/2);
	nbPosX = ceil((Max.x()-Min.x())/dx);
	nbPosZ = ceil((Max.z()-Min.z())/dz);
	uint size = nbPosX + nbPosX*nbPosZ;
	pos = new double[3*size];
	memset(pos, 0, 3*size*sizeof(double));
}
/*****************************************************************************************************/
void HeightField::create(Vector3 min, Vector3 max, double dx, double dz)
{
	this->Min = min;
	this->Max = max;
	this->dx = dx;
	this->dz = dz;
	this->center = Vector3((Max.x()+Min.x())/2,(Max.y()+Min.y())/2,(Max.z()+Min.z())/2);
	if(pos!=NULL)	delete[] pos;
	create();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
Vector3 HeightField::getCenter()
{
	return center;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
Vector3 HeightField::getMin()
{
	return Min;
}
/*****************************************************************************************************/
Vector3 HeightField::getMax()
{
	return Max;
}
/*****************************************************************************************************/
double HeightField::getDx()
{
	return dx;
}
/*****************************************************************************************************/
double HeightField::getDz()
{
	return dz;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void HeightField::generate()
{
	uint size = nbPosX + nbPosX*nbPosZ;
	if(pos==NULL)
		pos = new double[3*size];
	
	double* m_pos;
	allocateArray((void**)&m_pos, 3*size*sizeof(double));
	double sizeCellX = (Max.x()-Min.x())/(nbPosX-1);
	double sizeCellZ = (Max.z()-Min.z())/(nbPosZ-1);
	HeightField_initializePos_CUDA(center.x(),center.y(),center.z(),Max.x()-Min.x(),Max.z()-Min.z(),sizeCellX,sizeCellZ,nbPosX,nbPosZ,m_pos,size);
	calculateHeight(m_pos,size);
	copyArrayFromDevice(pos,m_pos, 0, 3*size*sizeof(double));
        threadSync();
	freeArray(m_pos);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void HeightField::display(Vector3 color)
{
	if(pos!=NULL){
		for(uint i=0;i<nbPosX-1;i++){
			for(uint j=0;j<nbPosZ-1;j++){
				uint index0 = i + j*nbPosX;
				double xC0 = pos[index0*3];
				double yC0 = pos[index0*3+1];
				double zC0 = pos[index0*3+2];
				uint index1 = i + (j+1)*nbPosX;
				double xC1 = pos[index1*3];
				double yC1 = pos[index1*3+1];
				double zC1 = pos[index1*3+2];
				uint index2 = (i+1) + (j+1)*nbPosX;
				double xC2 = pos[index2*3];
				double yC2 = pos[index2*3+1];
				double zC2 = pos[index2*3+2];
				uint index3 = (i+1) + j*nbPosX;
				double xC3 = pos[index3*3];
				double yC3 = pos[index3*3+1];
				double zC3 = pos[index3*3+2];
				glBegin(GL_TRIANGLES);
				glVertex3f(xC0,yC0,zC0);
				glVertex3f(xC1,yC1,zC1);
				glVertex3f(xC2,yC2,zC2);
				glEnd();
				glBegin(GL_TRIANGLES);
				glVertex3f(xC2,yC2,zC2);
				glVertex3f(xC3,yC3,zC3);
				glVertex3f(xC0,yC0,zC0);
				glEnd();
			}
		}
	}
}
/*****************************************************************************************************/
void HeightField::displayNormale(Vector3 color)
{
}
/*****************************************************************************************************/
/*****************************************************************************************************/
}
/*****************************************************************************************************/
/*****************************************************************************************************/	
