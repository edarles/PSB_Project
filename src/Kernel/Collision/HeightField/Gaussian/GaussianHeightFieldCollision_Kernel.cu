#include <GaussianHeightFieldCollision_Kernel.cuh>
#include <double3.h>
/******************************************************************************************/
/******************************************************************************************/
extern "C"
{
/******************************************************************************************/
/******************************************************************************************/
__host__ __device__ double  calculateHeight_Gaussian(double3 pos, double A, double x0, double z0, double p1, double p2)
{
	return A*exp(-((powf(pos.x-x0,2)/(2*p1*p1))+(powf(pos.z-z0,2)/(2*p2*p2))));	
}
/******************************************************************************************/
/******************************************************************************************/
__host__ __device__ double3 approximateNormale_Gaussian(double3 pos, double A, double x0, double z0, double p1, double p2)
{
	double d=0.01;
	double3 V1 = make_double3(pos.x+d,pos.y,pos.z);
	double3 V2 = make_double3(pos.x-d,pos.y,pos.z);
	double dpx = (calculateHeight_Gaussian(V1,A,x0,z0,p1,p2)-calculateHeight_Gaussian(V2,A,x0,z0,p1,p2))/(2*d);
	V1.x = pos.x; V1.z = pos.z+d; V2.x = pos.x; V2.z = pos.z-d;
	double dpy = 1.0;
	double dpz = (calculateHeight_Gaussian(V1,A,x0,z0,p1,p2)-calculateHeight_Gaussian(V2,A,x0,z0,p1,p2))/(2*d);
	double norme = sqrt(dpx*dpx + dpy*dpy + dpz*dpz);
	double3 N;
	N.x =-dpx/norme;
	N.y = dpy/norme;
	N.z =-dpz/norme;
	return N;
}

/******************************************************************************************/
/******************************************************************************************/
__global__ void collisionSystem_Gaussian_HeightFieldCollision_Kernel
	   (double3* newPos, double3 *newVel, double radiusParticle, float dt, uint nbBodiesP, 
	    double A, double x0, double z0, double p1, double p2,
	    float3 min_, float3 max_, float elast)
{
	
	int indexP = blockIdx.x * blockDim.x + threadIdx.x;
    	if(indexP < nbBodiesP){
		double3 pos = newPos[indexP];
		if(pos.x>=min_.x && pos.z>=min_.z && pos.x<=max_.x && pos.z<=max_.z){
			double y = calculateHeight_Gaussian(pos,A,x0,z0,p1,p2) + min_.y;
			if(y>=pos.y){
				double3 pInter = make_double3(pos.x,y,pos.z);
				float r = 0;
				if(length(newVel[indexP])>0 && elast>0 && length(newVel[indexP])>0)
		 			r = elast*length(pos - pInter)/(dt*length(newVel[indexP]));
				double3 nInter = approximateNormale_Gaussian(pInter,A,x0,z0,p1,p2);
				nInter = normalize(nInter);
               			double3 V;
				V.x = newVel[indexP].x - (1+r)*dot(newVel[indexP],nInter)*nInter.x;
				V.y = newVel[indexP].y - (1+r)*dot(newVel[indexP],nInter)*nInter.y;
				V.z = newVel[indexP].z - (1+r)*dot(newVel[indexP],nInter)*nInter.z;
				newVel[indexP] = make_double3(V.x,V.y,V.z);
				newPos[indexP].y = y;
			}
		}
	}
}
/******************************************************************************************/
/******************************************************************************************/
}
/******************************************************************************************/
/******************************************************************************************/
