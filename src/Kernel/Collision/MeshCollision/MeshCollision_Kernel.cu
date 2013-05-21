#include <math_constants.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <driver_types.h>
#include <cuda_gl_interop.h>	 
#include <stdio.h>
#include <host_defines.h>
#include <double3.h> 
#include <stdio.h> 
#include <AtomicDoubleAdd.cuh>

extern "C"
{
//**************************************************************************************************************************************
//**************************************************************************************************************************************
__host__ __device__ bool pointInTriangle(float3 A, float3 B, float3 C, float3 P)
{
	float3 u; u.x = B.x - A.x; u.y = B.y - A.y; u.z = B.z - A.z;
    	float3 v; v.x = C.x - A.x; v.y = C.y - A.y; v.z = C.z - A.z;
	float3 w; w.x = P.x - A.x; w.y = P.y - A.y; w.z = P.z - A.z;

	float    uu, uv, vv, wu, wv, D;
    	uu = u.x*u.x + u.y*u.y + u.z*u.z; //dot(u,u);
    	uv = u.x*v.x + u.y*v.y + u.z*v.z; //dot(u,v);
    	vv = v.x*v.x + v.y*v.y + v.z*v.z; //dot(v,v);

    	wu = w.x*u.x + w.y*u.y + w.z*u.z; //dot(w,u);
    	wv = w.x*v.x + w.y*v.y + w.z*v.z; //dot(w,v);
    	D = uv * uv - uu * vv;

    	// get and test parametric coords
    	float s, t;
    	s = (uv * wv - vv * wu) / D;
    	if (s < 0.0 || s > 1.0)         // I is outside T
        	return false;
    	t = (uv * wu - uu * wv) / D;
    	if (t < 0.0 || (s + t) > 1.0)  // I is outside T
        	return false;
   	return true;
}
//**************************************************************************************************************************************
//**************************************************************************************************************************************
__host__ __device__ bool detectionTriangle(double3 newPos, double3 oldPos, float3 A, float3 B, float3 C, float3 N, 
				           double3 *pt_int, double3 *nInter, double *distance)
{ 
	double3 P1 = make_double3(newPos.x-A.x, newPos.y-A.y, newPos.z-A.z);
	double d1 = P1.x*N.x + P1.y*N.y + P1.z*N.z;
	if(d1<=0){
		double3 AO = make_double3(A.x-newPos.x,A.y-newPos.y,A.z-newPos.z);
		double AO_N = AO.x*N.x + AO.y*N.y + AO.z*N.z;
		double3 dir = make_double3(oldPos.x-newPos.x,oldPos.y-newPos.y,oldPos.z-newPos.z);
		double dir_norm = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
		if(d1==0 || dir_norm<=0 || d1==-1){
		       pt_int->x = newPos.x;
		       pt_int->y = newPos.y;
		       pt_int->z = newPos.z;
		       return true;
		}
		dir.x = dir.x/dir_norm; dir.y = dir.y/dir_norm; dir.z = dir.z/dir_norm;
		double dir_N = dir.x*N.x + dir.y*N.y + dir.z*N.z;
		double t = AO_N / dir_N;
	
		pt_int->x = oldPos.x + t*dir.x;
		pt_int->y = oldPos.y + t*dir.y;
		pt_int->z = oldPos.z + t*dir.z;
		
		nInter->x = N.x;
		nInter->y = N.y;
		nInter->z = N.z;
		double3 D = make_double3(newPos.x-pt_int->x, newPos.y-pt_int->y, newPos.z-pt_int->z);
		(*distance) = sqrt(D.x*D.x + D.y*D.y + D.z*D.z);

		if(!pointInTriangle(A,B,C,make_float3(pt_int->x,pt_int->y,pt_int->z)))	return false;

		return true;       
	}		
	return false; 
}

//**************************************************************************************************************************************
//**************************************************************************************************************************************
// cuda collision routine
__global__ void collisionTriangle(uint nbBodies, uint nbFaces, double3* newPos, double3* newVel, 
          		    	  double3* oldPos, double3* oldVel, 
   			    	  float* AF, float* BF, float* CF, float* NF, float elast, float radiusParticle, float dt)
{
    int indexP = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(indexP < nbBodies){

    	double3 posAv = oldPos[indexP];
    	double3 velAv = oldVel[indexP];
    	double3 posAp = newPos[indexP];
    	double3 velAp = newVel[indexP];

	double3 PInter, NInter;
	double dMax = 10000000;
	bool trouve = false;

	for(int i=0;i<nbFaces;i++){ 
    		float3 A = make_float3(AF[(i*3)],AF[(i*3)+1],AF[(i*3)+2]);
    		float3 B = make_float3(BF[(i*3)],BF[(i*3)+1],BF[(i*3)+2]);
    		float3 C = make_float3(CF[(i*3)],CF[(i*3)+1],CF[(i*3)+2]);
    		float3 N = make_float3(NF[(i*3)],NF[(i*3)+1],NF[(i*3)+2]);

    		double3 pInter;
    		double3 nInter;
    		double distance;

    		//bool detect =  intersect(make_float3(posAv.x,posAv.y,posAv.z),make_float3(posAp.x,posAp.y,posAp.z),A,B,C,-N,&pInter);
		bool detect = detectionTriangle(posAp,posAv, A, B, C, N, &pInter, &nInter, &distance); 
				       
    		if(detect == true){
			nInter = make_double3(N.x,N.y,N.z);
			distance = sqrt(powf(posAp.x-pInter.x,2)+powf(posAp.y-pInter.y,2)+powf(posAp.z-pInter.z,2));
			if(distance<=dMax){
				dMax = distance;
				PInter = make_double3(pInter.x,pInter.y,pInter.z);
				NInter = make_double3(nInter.x,nInter.y,nInter.z);
				trouve = true;
			}
		}
	}
	if(trouve){
		float r = 0;
		elast = 0;
		if(length(velAp)>0 && elast>0 && length(velAp)>0)
			 r = elast*length(posAp - PInter)/(dt*length(velAp));
	        double3 V;
		V.x = velAp.x - (1+r)*dot(velAp,NInter)*NInter.x;
		V.y = velAp.y - (1+r)*dot(velAp,NInter)*NInter.y;
		V.z = velAp.z - (1+r)*dot(velAp,NInter)*NInter.z;
		newPos[indexP] = make_double3(PInter.x,PInter.y,PInter.z);
		newVel[indexP] = make_double3(V.x,V.y,V.z);
    	}
     }
}
}
