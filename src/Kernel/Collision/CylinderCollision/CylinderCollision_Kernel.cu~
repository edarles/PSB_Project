#include <CylinderCollision_Kernel.cuh>
#include <double3.h>

__global__ void collisionCylinder(int nbBodies, double3* newPos, double3* newVel, float radiusParticle, float dt,
			     	  float elast, bool container, float3 center, float baseRadius, float l, float3 direction)
{
	int indexP = blockIdx.x*blockDim.x + threadIdx.x;

    	if(indexP < nbBodies){

    		double3 posAp = newPos[indexP];
    		double3 velAp = newVel[indexP];
       		double3 pInter = make_double3(posAp.x,posAp.y,posAp.z);
    		double3 nInter = make_double3(0,0,0);
		bool collision = false;

		double3 P1 = make_double3(center.x + direction.x*(l/2),center.y + direction.y*(l/2),center.z + direction.z*(l/2));
		double3 P2 = make_double3(center.x - direction.x*(l/2),center.y - direction.y*(l/2),center.z - direction.z*(l/2));
		double3 C =  make_double3(center.x + posAp.x*direction.x, center.y + posAp.y*direction.y,center.z + posAp.z*direction.x);

		double dist = sqrt(powf(center.x-C.x,2)+powf(center.y-C.y,2)+powf(center.z-C.z,2));
		if(dist>=(l/2)){
			double dist1 = sqrt(powf(P1.x-C.x,2)+powf(P1.y-C.y,2)+powf(P1.z-C.z,2));
			double dist2 = sqrt(powf(P2.x-C.x,2)+powf(P2.y-C.y,2)+powf(P2.z-C.z,2));
			collision = true;
			if(dist1>dist2){
				double3 dir = make_double3(C.x-posAp.x,C.y-posAp.y,C.z-posAp.z);
				double lDir = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
				dir.x = dir.x/lDir; dir.y = dir.y/lDir; dir.z = dir.z/lDir;

				if(direction.x!=0)
					pInter.x = P2.x;
				if(direction.y!=0)
					pInter.y = P2.y;
				if(direction.z!=0)
					pInter.z = P2.z;
				if(lDir>=baseRadius){
					if(direction.x==0)
						pInter.x = C.x - (baseRadius-0.04)*dir.x;
					if(direction.y==0)
						pInter.y = C.y - (baseRadius-0.04)*dir.y;
					if(direction.z==0)
						pInter.z = C.z - (baseRadius-0.04)*dir.z;
				}
				nInter = make_double3(direction.x,direction.y,direction.z);
				double lnInter = sqrt(powf(nInter.x,2)+powf(nInter.y,2)+powf(nInter.z,2));
				nInter.x = nInter.x/lnInter; 
				nInter.y = nInter.y/lnInter; 
				nInter.z = nInter.z/lnInter; 
			}
			else {
				double3 dir = make_double3(C.x-posAp.x,C.y-posAp.y,C.z-posAp.z);
				double lDir = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
				dir.x = -dir.x/lDir; dir.y = -dir.y/lDir; dir.z = -dir.z/lDir;
				if(direction.x!=0)
					pInter.x = P1.x;
				if(direction.y!=0)
					pInter.y = P1.y;
				if(direction.z!=0)
					pInter.z = P1.z;
				if(lDir>=baseRadius){
					if(direction.x==0)
						pInter.x = C.x + (baseRadius-0.04)*dir.x;
					if(direction.y==0)
						pInter.y = C.y + (baseRadius-0.04)*dir.y;
					if(direction.z==0)
						pInter.z = C.z + (baseRadius-0.04)*dir.z;
				}
				nInter = make_double3(-direction.x,-direction.y,-direction.z);
				double lnInter = sqrt(powf(nInter.x,2)+powf(nInter.y,2)+powf(nInter.z,2));
				nInter.x = nInter.x/lnInter; 
				nInter.y = nInter.y/lnInter; 
				nInter.z = nInter.z/lnInter; 
			}
		}
		else {
			double dist1 = sqrt(powf(C.x-posAp.x,2)+powf(C.y-posAp.y,2)+powf(C.z-posAp.z,2));
			if(dist1>=baseRadius){
				collision = true;
				double3 dir = make_double3(C.x-posAp.x,C.y-posAp.y,C.z-posAp.z);
				double lDir = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
				dir.x = -dir.x/lDir; dir.y = -dir.y/lDir; dir.z = -dir.z/lDir;
				pInter.x = C.x + (baseRadius-0.04)*dir.x*(1-direction.x);
				pInter.y = C.y + (baseRadius-0.04)*dir.y*(1-direction.y);
				pInter.z = C.z + (baseRadius-0.04)*dir.z*(1-direction.z);
				nInter = dir;
			}
		}
		if(collision==true){ 
			newPos[indexP] = make_double3(pInter.x,pInter.y,pInter.z);
			float r = 0;
			double lV = sqrt(velAp.x*velAp.x+velAp.y*velAp.y+velAp.z*velAp.z);
			elast = 0.3;
			if(lV>0 && elast>0 && dt>0){
				double3 PI = make_double3(posAp.x-pInter.x,posAp.y-pInter.y,posAp.z-pInter.z);
				double lPI = sqrt(PI.x*PI.x + PI.y*PI.y + PI.z*PI.z);
				r = elast*lPI/(dt*lV);
			}
			double3 V;
			V.x = velAp.x - (1+r)*dot(velAp,nInter)*nInter.x;
			V.y = velAp.y - (1+r)*dot(velAp,nInter)*nInter.y;
			V.z = velAp.z - (1+r)*dot(velAp,nInter)*nInter.z;
			newVel[indexP] = make_double3(V.x,V.y,V.z);
		}
    }
}
