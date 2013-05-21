#include <SphKernel.cuh>
#include <double3.h>
#include <stdio.h>

extern "C" {
/******************************************************************************************************************/
// DENSITY AND PRESSURE EVALUATION CUDA ROUTINE
/******************************************************************************************************************/
__global__ void densityEvaluation(double3* pos, double* mass, double* radius, double* k, 
			          double* restDensity, uint nbBodies, double* density, double* pressure, partVoisine voisines)
{
	uint index1 =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index1<nbBodies){
		density[index1] = 0;
		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){
			int index2 = voisines.listeVoisine[(index1*200)+i];			
			double3 P1P2 = pos[index1] - pos[index2];
			double d = length(P1P2);
			double a = 64*M_PI*pow(radius[index2],9);
			double m_k = 315/a;
			double b = pow((radius[index2]*radius[index2])-(d*d),3);
			density[index1] += b*m_k*mass[index2];
			
		}
		pressure[index1] = k[index1]*(density[index1] - restDensity[index1]);
	}
}

/******************************************************************************************************************/
// PRESSURE, VISCOSITY AND SURFACE TENSION FORCES EVALUATION CUDA ROUTINE
/******************************************************************************************************************/
__global__ void internalForces(double3* pos, double3* vel, double* mass, double* density, double* pressure, 
			       double* radius, double* viscosity, double* l, double* surfaceTension, double3* normales, uint nbBodies,
			       double3* forcePressure, double3* forceViscosity, double3* forceSurface, double3* forcesAccum, 
			       partVoisine voisines)
{

        uint index1 =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
        if(index1<nbBodies){

		forcePressure[index1] = make_double3(0,0,0);
		forceViscosity[index1] = make_double3(0,0,0);
		normales[index1] = make_double3(0,0,0);
		double fSurface = 0;

		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){

			int index2 = voisines.listeVoisine[(index1*200)+i];
			double3 P1P2 = pos[index1] - pos[index2];
			double d = length(P1P2);

			if(d>0)
			{
				// Pression force
				double3 P1P2_N = P1P2/d;
				double pV1 = pressure[index1]/(density[index1]*density[index1]);
				double pV2 = pressure[index2]/(density[index2]*density[index2]);
				double a = M_PI*pow(radius[index2],6);
				double3 m_k = P1P2_N*(-45*pow(radius[index2]-d,2)/a);;				
				forcePressure[index1] = forcePressure[index1] + m_k*(pV1+pV2)*mass[index2];
	
                		// Viscosity force
				double  m_m = 45*(radius[index2]-d)/a;
				forceViscosity[index1] = forceViscosity[index1] + 
				((viscosity[index1]+viscosity[index2])/2)*(vel[index2]-vel[index1])*mass[index2]*m_m/density[index2];
			
				// surface tension force
				double b = 32*M_PI*powf(radius[index2],9);
				double m_d = -(945/b)*((radius[index2]*radius[index2])-(d*d))*((3*radius[index2]*radius[index2])-(7*d*d));
				fSurface += m_d*(mass[index2]/density[index2]);

				// normale evaluation
				double mk = -945/b;
				normales[index1] = normales[index1] + P1P2*(mass[index2]/density[index2])*
				pow((radius[index2]*radius[index2])-(d*d),2)*mk;
			}
		}
		forcePressure[index1] = forcePressure[index1]*(-density[index1]);
		double lNi = length(normales[index1]);
		if(lNi>=l[index1]){
			forceSurface[index1] = fSurface*(-surfaceTension[index1])*normales[index1]/lNi;
			normales[index1] = normalize(normales[index1]);
		}
		else {
			normales[index1] = make_double3(0,0,0);
			forceSurface[index1] = make_double3(0,0,0);
		}
		forcesAccum[index1] = forceViscosity[index1] + forceSurface[index1] + forcePressure[index1];
	}
}

/******************************************************************************************************************/
/******************************************************************************************************************/
__global__ void integrateSPH_LeapFrog(double3* velInterAv, double3* velInterAp, double3* oldPos, double3* newPos, 
				      double3* forces, double*  densities, float dt, uint nbBodies)
{
   uint index =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   if(index<nbBodies){
		// LEAP-FROG SCHEME (see [Kelager06])
		velInterAv[index] = velInterAp[index];
		velInterAp[index] = velInterAp[index] + (forces[index]/densities[index])*dt; 
		oldPos[index] = newPos[index];
		newPos[index] = newPos[index] + velInterAp[index]*dt;

   }
}

/******************************************************************************************************************/
/******************************************************************************************************************/
__global__ void interpolateSPH_velocities(double3* velInterAv, double3* velInterAp, double3* oldVel, double3* newVel, uint nbBodies)
{
   uint index =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   if(index<nbBodies){
	oldVel[index] = newVel[index];
	newVel[index] = (velInterAv[index] + velInterAp[index])/2;
   }
}
/******************************************************************************************************************/
/******************************************************************************************************************/
}
/******************************************************************************************************************/
/******************************************************************************************************************/


