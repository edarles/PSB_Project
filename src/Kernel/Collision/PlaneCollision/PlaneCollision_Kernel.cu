#include <PlaneCollision_Kernel.cuh>
#include <double3.h>

//**************************************************************************************************************************************
//**************************************************************************************************************************************
__host__ __device__ bool detectionPlan(double3 oldPos, double3 newPos, float3 A, float3 B, float3 C, 
				       float3 D, float3 N, double3 *pt_int, double3 *nInter, float *distance)
{ 
  float3 MIN = make_float3(100000,100000,100000);
  float3 MAX = make_float3(-100000,-100000,-100000);

  if(MIN.x>=A.x) MIN.x = A.x; if(MIN.x>=B.x) MIN.x = B.x; if(MIN.x>=C.x) MIN.x = C.x; if(MIN.x>=D.x) MIN.x = D.x;
  if(MIN.y>=A.y) MIN.y = A.y; if(MIN.y>=B.y) MIN.y = B.y; if(MIN.y>=C.y) MIN.y = C.y; if(MIN.y>=D.y) MIN.y = D.y;
  if(MIN.z>=A.z) MIN.z = A.z; if(MIN.z>=B.z) MIN.z = B.z; if(MIN.z>=C.z) MIN.z = C.z; if(MIN.z>=D.z) MIN.z = D.z;

  if(MAX.x<=A.x) MAX.x = A.x; if(MAX.x<=B.x) MAX.x = B.x; if(MAX.x<=C.x) MAX.x = C.x; if(MAX.x<=D.x) MAX.x = D.x;
  if(MAX.y<=A.y) MAX.y = A.y; if(MAX.y<=B.y) MAX.y = B.y; if(MAX.y<=C.y) MAX.y = C.y; if(MAX.y<=D.y) MAX.y = D.y;
  if(MAX.z<=A.z) MAX.z = A.z; if(MAX.z<=B.z) MAX.z = B.z; if(MAX.z<=C.z) MAX.z = C.z; if(MAX.z<=D.z) MAX.z = D.z;

  if(newPos.x*(1-N.x)<=MAX.x*(1-N.x) && newPos.x*(1-N.x)>=MIN.x*(1-N.x) && newPos.y*(1-N.y)<=MAX.y*(1-N.y) && newPos.y*(1-N.y)>=MIN.y*(1-N.y) && newPos.z*(1-N.z)<=MAX.z*(1-N.z) && newPos.z*(1-N.z)>=MIN.z*(1-N.z)){
	float3 P1 = make_float3(newPos.x-A.x, newPos.y-A.y, newPos.z-A.z);
	float d1 = P1.x*N.x + P1.y*N.y + P1.z*N.z;
	if(d1<=0){
		float3 AO = make_float3(A.x-oldPos.x,A.y-oldPos.y,A.z-oldPos.z);
		float AO_N = AO.x*N.x + AO.y*N.y + AO.z*N.z;
		float3 dir = make_float3(newPos.x-oldPos.x,newPos.y-oldPos.y,newPos.z-oldPos.z);
		float dir_norm = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
		dir.x = dir.x/dir_norm; dir.y = dir.y/dir_norm; dir.z = dir.z/dir_norm;
		float dir_N = dir.x*N.x + dir.y*N.y + dir.z*N.z;
		float t = AO_N / dir_N;
		pt_int->x = oldPos.x + t*dir.x;
		pt_int->y = oldPos.y + t*dir.y;
		pt_int->z = oldPos.z + t*dir.z;
		
		nInter->x = N.x;
		nInter->y = N.y;
		nInter->z = N.z;
		float3 D = make_float3(newPos.x-pt_int->x, newPos.y-pt_int->y, newPos.z-pt_int->z);
		(*distance) = sqrt(D.x*D.x + D.y*D.y + D.z*D.z);
		return true;
	}
	return false;
  }
  return false;
}

//**************************************************************************************************************************************
//**************************************************************************************************************************************
// cuda collision routine
__global__ void collisionPlan(uint nbBodies, double3* newPos, double3* newVel, 
          		      double3* oldPos, double3* oldVel, 
   			      float3 A, float3 B, float3 C, float3 D, float3 N, float elast, float dt)
{
    int indexP = blockIdx.x * blockDim.x + threadIdx.x;
   
    if(indexP<nbBodies){

    	double3 posAv = oldPos[indexP];
    	double3 posAp = newPos[indexP];
    	double3 velAp = newVel[indexP];
 
    	double3 pInter;
        double3 nInter;
        float distance;
  
    	bool detect = detectionPlan(posAv,posAp,A,B,C,D,N,&pInter,&nInter,&distance);
    	if(detect == true) {
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

