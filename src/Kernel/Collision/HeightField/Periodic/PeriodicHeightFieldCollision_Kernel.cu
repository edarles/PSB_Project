#include <PeriodicHeightFieldCollision_Kernel.cuh>
#include <double3.h>
/******************************************************************************************/
/******************************************************************************************/
extern "C"
{
/******************************************************************************************/
/******************************************************************************************/
__host__ __device__ double  calculateHeight_Periodic(double3 pos, uint nbFunc, double* A, double* k, double* theta, double* phi)
{
	double y = 0;
	for(uint i=0;i<nbFunc;i++)
		y+= A[i]*cos(k[i]*(pos.x*cos(theta[i])+pos.z*sin(theta[i]))+phi[i]);
	return y;
}
/******************************************************************************************/
/******************************************************************************************/
__host__ __device__ double3 approximateNormale_Periodic(double3 pos, uint nbFunc, double* A, double* k, double* theta, double* phi)
{
	double d=0.01;
	double3 V1 = make_double3(pos.x+d,pos.y,pos.z);
	double3 V2 = make_double3(pos.x-d,pos.y,pos.z);
	double dpx = (calculateHeight_Periodic(V1,nbFunc,A,k,theta,phi)-calculateHeight_Periodic(V2,nbFunc,A,k,theta,phi))/(2*d);
	V1.x = pos.x; V1.z = pos.z+d; V2.x = pos.x; V2.z = pos.z-d;
	double dpy = 1.0;
	double dpz = (calculateHeight_Periodic(V1,nbFunc,A,k,theta,phi)-calculateHeight_Periodic(V2,nbFunc,A,k,theta,phi))/(2*d);
	double norme = sqrt(dpx*dpx + dpy*dpy + dpz*dpz);
	double3 N;
	N.x =-dpx/norme;
	N.y = dpy/norme;
	N.z =-dpz/norme;
	return N;
}

/******************************************************************************************/
/******************************************************************************************/
__global__ void collisionSystem_Periodic_HeightFieldCollision_Kernel
	   (double3* newPos, double3 *newVel, double radiusParticle, float dt, uint nbBodiesP, 
	    uint nbFunc, double* A, double* k, double* theta, double* phi,
	    float3 min_, float3 max_, float elast)
{
	
	int indexP = blockIdx.x * blockDim.x + threadIdx.x;
    	if(indexP < nbBodiesP){
		double3 pos = newPos[indexP];
		if(pos.x>=min_.x && pos.z>=min_.z && pos.x<=max_.x && pos.z<=max_.z){
			double y = calculateHeight_Periodic(pos,nbFunc,A,k,theta,phi) + min_.y;
			if(y>=pos.y){
				double3 pInter = make_double3(pos.x,y,pos.z);
				float r = 0;
				if(length(newVel[indexP])>0 && elast>0 && length(newVel[indexP])>0)
		 			r = elast*length(pos - pInter)/(dt*length(newVel[indexP]));
				double3 nInter = approximateNormale_Periodic(pInter,nbFunc,A,k,theta,phi);
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
