#include <GaussianHeightField_Kernel.cuh>
/****************************************************************************************************************************/
/****************************************************************************************************************************/
extern "C"
{
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void GaussianHeightField_calculateHeight_Kernel(double3* pos, double A, double x0, double z0, double p1, double p2, uint nbPos)
{
	uint index =   __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if(index < nbPos){
		pos[index].y += A*exp(-((powf(pos[index].x-x0,2)/(2*p1*p1))+(powf(pos[index].z-z0,2)/(2*p2*p2))));
	}		
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void GaussianHeightField_calculateHeight_Normales_Kernel(double3* m_pos, double3* nx0, double3* nx1,
								   double3* nz0, double3* nz1, double A, double x0, double z0, 
								   double p1, double p2, uint nbPos)
{
	uint index =   __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if(index < nbPos){
		m_pos[index].y += A*exp(-((powf(m_pos[index].x-x0,2)/(2*p1*p1))+(powf(m_pos[index].z-z0,2)/(2*p2*p2))));
		nx0[index].y += A*exp(-((powf(nx0[index].x-x0,2)/(2*p1*p1))+(powf(nx0[index].z-z0,2)/(2*p2*p2))));
		nx1[index].y += A*exp(-((powf(nx1[index].x-x0,2)/(2*p1*p1))+(powf(nx1[index].z-z0,2)/(2*p2*p2))));
		nz0[index].y += A*exp(-((powf(nz0[index].x-x0,2)/(2*p1*p1))+(powf(nz0[index].z-z0,2)/(2*p2*p2))));
		nz1[index].y += A*exp(-((powf(nz1[index].x-x0,2)/(2*p1*p1))+(powf(nz1[index].z-z0,2)/(2*p2*p2))));
	}	
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
