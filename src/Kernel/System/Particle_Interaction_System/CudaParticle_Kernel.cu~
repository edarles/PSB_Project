#include <stdio.h>
#include <CudaParticle_Kernel.cuh>
#include <double3.h>

extern "C" 
{
/****************************************************************************************************/
// collide two spheres using DEM method
/****************************************************************************************************/
__device__ double3 collideParticle(double3 posA, double3 posB,
                                  double3 velA, double3 velB,
                                  double radiusA, double radiusB,
                                  double spring, double damping, double shear, double attraction)
{
   // calculate relative position
    double3 relPos = posB - posA;
    double dist = length(relPos);
    double3 force = make_double3(0,0,0);

    if(dist>0){
    	float collideDist = (radiusA + radiusB);
    	if (dist < collideDist*2) {
        	double3 norm = relPos / dist;
		// relative velocity
        	double3 relVel = velB - velA;
       		double3 tanVel = relVel - dot(relVel, norm) * norm;
        	// spring force
        	force = -spring*(collideDist - dist) * norm;
        	// dashpot (damping) force
        	force += damping*relVel;
        	// tangential shear force
        	force += shear*tanVel;
		// attraction
        	force += attraction*relPos;
    	}
    }
    return force;
}

/****************************************************************************************************/
/****************************************************************************************************/
__global__ void collideParticles(uint nbBodies, double3* oldPos, double3* oldVel,
				 double3* forces, double *radius, double *spring, 
				 double *damping, double* shear, double* attraction, partVoisine voisines)
{
    int indexA = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if(indexA < nbBodies){
	    for(unsigned int i=0;i<voisines.nbVoisines[indexA];i++){
			int indexB = voisines.listeVoisine[(indexA*200)+i];
 	    		forces[indexA] = forces[indexA] + collideParticle(oldPos[indexA],oldPos[indexB],oldVel[indexA],oldVel[indexB],
						        radius[indexA],radius[indexB],
					                spring[indexA],damping[indexA],shear[indexA],attraction[indexA]);
	    }
    }
}
/****************************************************************************************************/
/****************************************************************************************************/
}
