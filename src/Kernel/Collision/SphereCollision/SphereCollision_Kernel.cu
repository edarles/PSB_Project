#include <common.cuh>

#include <double3.h> 

//**************************************************************************************************************************************
//**************************************************************************************************************************************
__host__ __device__ bool detection_collisionSphere(double3 oldPos, double3 newPos, float3 origin, float radius, 
					           bool container, double3 *pt_int, double3 *nInter, double *distance)
{ 
  double3 OP = make_double3(newPos.x-origin.x,newPos.y-origin.y,newPos.z-origin.z);
  double  dOP = length(OP);
  if(container && dOP>=radius)
  {
    if(dOP == radius){
	(*pt_int) = newPos;
	(*nInter) = -OP/dOP;
	(*distance) = 0;
	return true;
    }
    if(dOP>radius){
    	double3 NOP = OP/dOP;
    	(*pt_int) = origin + NOP*radius;
    	(*distance) = fabs(radius - dOP);
    	(*nInter) = origin - (*pt_int);
    	(*nInter) = (*nInter)/length(*nInter);
    	return true;
    }
    return false;
  }
  if(!container && dOP<=radius)
  {
	 if(dOP == radius){
	(*pt_int) = newPos;
	(*nInter) = -OP/dOP;
	(*distance) = 0;
	return true;
    }
    if(dOP>radius){
    	double3 NOP = -OP/dOP;
    	(*pt_int) = origin + NOP*radius;
    	(*distance) = fabs(radius - dOP);
    	(*nInter) = origin - (*pt_int);
    	(*nInter) = (*nInter)/length(*nInter);
    	return true;
    }
    return false;
  }
  return false;
}

//**************************************************************************************************************************************
//**************************************************************************************************************************************
// cuda collision routine
__global__ void collisionSphere(uint nbBodies, double3* newPos, double3* newVel, double3* oldPos, double3* oldVel,
				float3  origin, float radius, float elast, bool container, float dt)
{
    int indexP = blockIdx.x * blockDim.x + threadIdx.x;

    if(indexP < nbBodies){

    	double3 posAv = oldPos[indexP];
    	double3 posAp = newPos[indexP];
    	double3 velAp = newVel[indexP];

    	double3 pInter,nInter;
    	double distance;
   
    	bool detect = detection_collisionSphere(posAv,posAp,origin,radius,container,&pInter,&nInter,&distance);
    	if(detect == true){
		newPos[indexP] = make_double3(pInter.x,pInter.y,pInter.z);
		float r = 0;
		if(length(velAp)>0 && elast>0 && length(velAp)>0)
		 	r = elast*length(posAp - pInter)/(dt*length(velAp));
		nInter = normalize(nInter);
                double3 V;
		V.x = velAp.x - (1+r)*dot(velAp,nInter)*nInter.x;
		V.y = velAp.y - (1+r)*dot(velAp,nInter)*nInter.y;
		V.z = velAp.z - (1+r)*dot(velAp,nInter)*nInter.z;
		newVel[indexP] = make_double3(V.x,V.y,V.z);
    	}
    }
}

