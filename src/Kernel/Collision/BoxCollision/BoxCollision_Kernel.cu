#include <stdio.h> 
#include <double3.h>

//**************************************************************************************************************************************
//**************************************************************************************************************************************
// cuda collision routine
__global__ void collisionBox(int nbBodies, double3* newPos, double3* oldVel, double3* newVel, float3 Min, float3 Max,
			     float elast, bool container, float radiusParticles, float dt)
{
    int indexP = blockIdx.x*blockDim.x + threadIdx.x;

    if(indexP < nbBodies){

    	double3 pInter = newPos[indexP];
    	double3 nInter = make_double3(0,0,0);

  	if(newPos[indexP].x >= Max.x-radiusParticles) { pInter.x = Max.x-radiusParticles; nInter.x = -1.0; }
	if(newPos[indexP].y >= Max.y-radiusParticles) { pInter.y = Max.y-radiusParticles; nInter.y = -1.0; }
	if(newPos[indexP].z >= Max.z-radiusParticles) { pInter.z = Max.z-radiusParticles; nInter.z = -1.0; }
	
	if(newPos[indexP].x <= Min.x+radiusParticles) { pInter.x = Min.x+radiusParticles; nInter.x = 1.0; }
	if(newPos[indexP].y <= Min.y+radiusParticles) { pInter.y = Min.y+radiusParticles; nInter.y = 1.0; }
	if(newPos[indexP].z <= Min.z+radiusParticles) { pInter.z = Min.z+radiusParticles; nInter.z = 1.0; }
   
    	if(length(nInter)>0){
		nInter = normalize(nInter);
		double r = 0;
		//if(length(newVel[indexP])>0)
		//	r = elast*length(newPos[indexP] - pInter)/(dt*length(newVel[indexP]));
		newPos[indexP] = pInter;
		newVel[indexP] = newVel[indexP] - (1+r)*dot(newVel[indexP],nInter)*nInter;
    }
   }
}
