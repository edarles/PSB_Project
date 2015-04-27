#include <AnimatedPeriodicHeightField_Kernel.cuh>
#include <stdio.h>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void AnimatedPeriodicHeightField_calculateHeight_Kernel(double3* pos, double* A, double* k, double* theta, 
								   double* phi ,double *omega, double t, uint nbFunc, uint nbPos)
{
    uint index =  __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if(index < nbPos){
	double y = 0;
	for(uint i=0;i<nbFunc;i++)
		y += A[i]*cos(k[i]*(pos[index].x*cos(theta[i])+pos[index].z*sin(theta[i]))-omega[i]*t+phi[i]);
	pos[index].y = y;
    }	
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void AnimatedPeriodicHeightField_calculateHeight_Normales_Gradient_Kernel(double3* m_pos, double3* m_N, double* A, double* k, 
									    	     double* theta, double* phi,
									             double *omega, double t, uint nbFunc, uint nbPos)
{
	uint index =   __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if(index < nbPos){
		double x = 0;
		double z = 0;
		for(uint i=0;i<nbFunc;i++){
			x -= A[i]*k[i]*cos(theta[i])*sin(k[i]*(m_pos[index].x*cos(theta[i])+m_pos[index].z*sin(theta[i]))-omega[i]*t+phi[i]);
			z -= A[i]*k[i]*sin(theta[i])*sin(k[i]*(m_pos[index].x*cos(theta[i])+m_pos[index].z*sin(theta[i]))-omega[i]*t+phi[i]);
		}
		m_N[index].x = x;
		m_N[index].y = 1;
		m_N[index].z = z;
	}
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
