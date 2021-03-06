#include <PeriodicHeightField_Kernel.cuh>
#include <stdio.h>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void PeriodicHeightField_calculateHeight_Kernel(double3* pos, double* A, double* k, double* theta, double* phi, uint nbFunc, uint nbPos)
{
	uint index =  __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if(index < nbPos){
		double y = 0;
		for(uint i=0;i<nbFunc;i++)
			y += A[i]*cos(k[i]*(pos[index].x*cos(theta[i])+pos[index].z*sin(theta[i]))+phi[i]);
		pos[index].y += y;
	}		
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void PeriodicHeightField_calculateHeight_Normales_Kernel(double3* m_pos, double3* nx0, double3* nx1, 
			  	double3* nz0, double3* nz1, double* A, double* k, double* theta, double* phi, uint nbFunc, uint nbPos)
{
	uint index =   __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if(index < nbPos){
		for(uint i=0;i<nbFunc;i++){
			m_pos[index].y += A[i]*cos(k[i]*(m_pos[index].x*cos(theta[i])+m_pos[index].z*sin(theta[i]))+phi[i]);
			nx0[index].y += A[i]*cos(k[i]*(nx0[index].x*cos(theta[i])+nx1[index].z*sin(theta[i]))+phi[i]);
			nx1[index].y += A[i]*cos(k[i]*(nx1[index].x*cos(theta[i])+nx0[index].z*sin(theta[i]))+phi[i]);
			nz0[index].y += A[i]*cos(k[i]*(nz0[index].x*cos(theta[i])+nz0[index].z*sin(theta[i]))+phi[i]);
			nz1[index].y += A[i]*cos(k[i]*(nz1[index].x*cos(theta[i])+nz1[index].z*sin(theta[i]))+phi[i]);
		}
	}	
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
