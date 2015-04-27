#include <stdio.h> 
#include <double3.h>

//**************************************************************************************************************************************
//**************************************************************************************************************************************
// cuda collision routine (3D version)
__global__ void collisionBox(int nbBodies, double3* newPos, double3* oldPos, double3* oldVel, double3* newVel, float3 Min, float3 Max,
			     float elast, bool container, float radiusParticles, float dt)
{
   int indexP = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if(indexP < nbBodies){

    	double3 pInter = newPos[indexP];
    	double3 nInter = make_double3(0,0,0);
  	if (newPos[indexP].x<Min.x+radiusParticles) nInter.x = 1.0;
	if (newPos[indexP].y<Min.y+radiusParticles) nInter.y = 1.0;
	if (newPos[indexP].z<Min.z+radiusParticles) nInter.z = 1.0;

	if (newPos[indexP].x>Max.x-radiusParticles) nInter.x = -1.0;
	if (newPos[indexP].y>Max.y-radiusParticles) nInter.y = -1.0;
	if (newPos[indexP].z>Max.z-radiusParticles) nInter.z = -1.0;

    	if(length(nInter)>0){

		pInter.x = fmax((float)pInter.x,(float)(Min.x+radiusParticles));
	        pInter.x = fmin((float)pInter.x,(float)(Max.x-radiusParticles));
		pInter.y = fmax((float)pInter.y,(float)(Min.y+radiusParticles)); 
		pInter.y = fmin((float)pInter.y,(float)(Max.y-radiusParticles));
		pInter.z = fmax((float)pInter.z,(float)(Min.z+radiusParticles)); 
		pInter.z = fmin((float)pInter.z,(float)(Max.z-radiusParticles));
/*
		double r = 0;
		double3 vel = newVel[indexP];
		if(length(vel)>0)
		 	r = 0.1*length(newPos[indexP] - pInter)/(dt*length(vel));
		nInter = normalize(nInter);

		oldVel[indexP] = newVel[indexP];
                newVel[indexP] = vel - (1+r)*dot(vel,nInter)*nInter;
		oldPos[indexP] = newPos[indexP];
		newPos[indexP] = pInter + newVel[indexP]*dt;

*/		nInter = normalize(nInter);
		double3 vel = make_double3(newVel[indexP].x,newVel[indexP].y,newVel[indexP].z);
		double3 vN = dot(vel,nInter)*nInter;
		double3 vT = vel - vN;
		float fric = 0.1; elast = 0.6;
		vel = (1-fric)*vel - (1+elast)*vN;
		oldPos[indexP] = newPos[indexP];
		newVel[indexP] = vel;
		newPos[indexP] = pInter + newVel[indexP]*dt;
/*
		nInter = normalize(nInter);
		double3 vel = make_double3(newVel[indexP].x,newVel[indexP].y,newVel[indexP].z);
		vel = vel - 0.5*(2*radiusParticles-length(newPos[indexP] - pInter))*nInter;//(1-elast)*vT - elast*vN; 
		oldVel[indexP] = newVel[indexP];
		oldPos[indexP] = newPos[indexP];
		newVel[indexP] = vel;
		newPos[indexP] = pInter;
*/		
       }
   }
}

//**************************************************************************************************************************************
//**************************************************************************************************************************************
// cuda collision routine (2D version)
__global__ void collisionBox2D(int nbBodies, double3* newPos, double3* oldVel, double3* newVel, float3 Min, float3 Max, 
			       float elast, bool container, float radiusParticles, float dt)
{
   int indexP = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if(indexP < nbBodies){
    	double3 pInter = newPos[indexP];
    	double3 nInter = make_double3(0,0,0);
  	nInter.x = (newPos[indexP].x<Min.x+radiusParticles);
	nInter.y = (newPos[indexP].y<Min.y+radiusParticles);
	nInter.x += -(newPos[indexP].x>Max.x-radiusParticles);
	nInter.y += -(newPos[indexP].y>Max.y-radiusParticles);
	
    	if(length(nInter)>0){

		pInter.x = fmax((float)pInter.x,(float)(Min.x+radiusParticles)); 
		pInter.x = fmin((float)pInter.x,(float)(Max.x-radiusParticles));
		pInter.y = fmax((float)pInter.y,(float)(Min.y+radiusParticles)); 
		pInter.y = fmin((float)pInter.y,(float)(Max.y-radiusParticles));

		nInter = normalize(nInter);
		double3 vel = make_double3(newVel[indexP].x,newVel[indexP].y,0);
		double3 vN = dot(vel,nInter)*nInter;
		double3 vT = vel - vN;
		vel = (1-elast)*vT - elast*vN; //(1-elast)*vT + elast*vN;
		newVel[indexP] = vel;
		newPos[indexP] = pInter;
    }
}

}
