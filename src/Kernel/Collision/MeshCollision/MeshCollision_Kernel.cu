#include <double3.h> 
#include <stdio.h> 

#include <common.cuh>

//**************************************************************************************************************************************
//**************************************************************************************************************************************
extern "C"
{
//**************************************************************************************************************************************
//**************************************************************************************************************************************
__global__ void collisionTriangle(uint nbBodies, uint nbFaces, double3* newPos, double3* newVel, 
          		    	  double3* oldPos, double3* oldVel, 
   			    	  float* AF, float* BF, float* CF, float* NF, float elast, float radiusParticle, float dt)
{
    int indexP = blockIdx.x * blockDim.x + threadIdx.x;

    if(indexP<nbBodies){

    	double3 posAv = oldPos[indexP];
    	double3 posAp = newPos[indexP];
    	double3 velAp = newVel[indexP];

	double3 NInter, PInter;
	double dMax = 10000;
	bool trouve = false;

	for(uint indexF=0;indexF<nbFaces;indexF++){
    		double3 A = make_double3(AF[(indexF*3)],AF[(indexF*3)+1],AF[(indexF*3)+2]);
    		double3 B = make_double3(BF[(indexF*3)],BF[(indexF*3)+1],BF[(indexF*3)+2]);
    		double3 C = make_double3(CF[(indexF*3)],CF[(indexF*3)+1],CF[(indexF*3)+2]);
    		double3 N = make_double3(NF[(indexF*3)],NF[(indexF*3)+1],NF[(indexF*3)+2]);

		double d1 = dot(posAp-A,N);
		if(d1<=0){
			double3 X;
			double d0 = dot(posAv-A,N); 
			if((d0-d1)!=0)
				X = ((d0*posAp) - (d1*posAv))/(d0-d1);
			else 
				X = posAp;
			double t1 = dot(cross(B-A,X-A),cross(X-A,C-A));
			double t2 = dot(cross(A-B,X-B),cross(X-B,C-B));
			double t3 = dot(cross(A-C,X-C),cross(X-C,B-C));
			if(t1>=0 || t2>=0 || t3>=0){
				double dist = length(posAv-X);
				if(dist<=dMax){
					dMax = dist;
					trouve = true;
					PInter = X;
					NInter = N;
				}
			}
		}
	}
	if(trouve==true){
		newPos[indexP] = PInter;
		double3 vN = dot(velAp,NInter)*NInter*elast;
		double3 vT = (velAp - vN)*(1-elast);
		newVel[indexP] = vT - vN;
	}
     }
}
//**************************************************************************************************************************************
//**************************************************************************************************************************************
}
//**************************************************************************************************************************************
//**************************************************************************************************************************************
