#include <SphKernel.cuh>
#include <double3.h>
#include <AtomicDoubleAdd.cuh>
#include <stdio.h>
#include <Textures.cuh>

extern "C" {

/******************************************************************************************************************/
// DENSITY AND PRESSURE EVALUATION CUDA ROUTINE
// WITH MEMORY TEXTURE - NE MARCHE PAS EN L'ETAT
/******************************************************************************************************************/
/******************************************************************************************************************/
__global__ void densityEvaluation_Texture(double3* pos,  uint nbBodies, double* density, double* pressure, partVoisine voisines)
{
	uint index1 =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index1<nbBodies){
		double dens = 0;
		double3 P1 = pos[index1];
		printf("index1:%d mass:%f\n",index1,fetch_double(texMass,index1));
		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){
			int index2 = voisines.listeVoisine[(index1*200)+i];			
			double3 P1P2 = P1 - pos[index2];
			double r = fetch_double(getDeviceTexRadius(),index2);
			if(r!=0) printf("r:%f\n",r);
			
			double d = length(P1P2);
			double a = 64*M_PI*pow(r,9);
			double m_k = 315/a;
			double b = pow((r*r)-(d*d),3);
			double m = fetch_double(getDeviceTexMass(),index2);
			if(m!=0) printf("m:%f\n",m);
			dens += b*m_k*m;
			
		}
		density[index1] = dens;
		double k = fetch_double(getDeviceTexK(),index1);
		double rho = fetch_double(getDeviceTexRestDensities(),index1);
		printf("k:%f\n",k);
		printf("rho:%f\n",rho);
		pressure[index1] = k*(dens - rho);
		printf("dens:%f p:%f\n",density[index1],pressure[index1]);
	}
}

/******************************************************************************************************************/
// DENSITY AND PRESSURE EVALUATION CUDA ROUTINE
/******************************************************************************************************************/
/******************************************************************************************************************/
__global__ void densityEvaluation(double3* pos,  uint nbBodies, double* radius, double* mass, double* k, double* rho0,
			          double* density, double* pressure, partVoisine voisines)
{
	uint index1 =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index1<nbBodies){
		double dens = 0;
		double3 P1 = pos[index1];
		double rI = radius[index1];
		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){
			int index2 = voisines.listeVoisine[(index1*200)+i];			
			double3 P1P2 = P1 - pos[index2];
			double r = max(radius[index2],rI);
			double d = length(P1P2);
			double a = 64*M_PI*pow(r,9);
			double m_k = 315/a;
			double b = pow((r*r)-(d*d),3);
			double m = mass[index2];
			dens += b*m_k*m;
		}
		density[index1] = dens;
		double kI = k[index1];
		double rho = rho0[index1];
		pressure[index1] = 1119.0*(powf(dens/rho,7)-1);//kI*(dens - rho);
		//if(voisines.nbVoisines[index1]>60) printf("nb:%d\n",voisines.nbVoisines[index1]);
		//printf("dens:%f p:%f\n",density[index1],pressure[index1]);
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

		register double3 fP = make_double3(0,0,0);
		register double3 fV = make_double3(0,0,0);
		register double3 N = make_double3(0,0,0);
		register double fS = 0;

		// attributes declaration;
		double d1 = density[index1];
		double p1 = pressure[index1];
		double pV1 = p1/(d1*d1);
		double v1 = viscosity[index1];
		double3 vel1 = vel[index1];
		double r1 = radius[index1];

		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){

			int index2 = voisines.listeVoisine[(index1*200)+i];
			double3 P1P2 = pos[index1] - pos[index2];
			double d = length(P1P2);

			if(d>0) {			
				// attributes declaration
				double d2 = density[index2];
				double p2 = pressure[index2];
				double pV2 = p2/(d2*d2);
				double r2 = max(radius[index2],r1);
				//if(d<=r2){
				double m2 = mass[index2];
				double v2 = viscosity[index2];
				double3 vel2 = vel[index2];
				double a = M_PI*pow(r2,6);

				// Pression force
				double3 P1P2_N = P1P2/d;
				
				double3 m_k = P1P2_N*(-45*pow(r2-d,2)/a);			
				fP =  fP + m_k*(pV1+pV2)*m2;
				
                		// Viscosity force
				double  m_m = 45*(r2-d)/a;
				fV = fV + ((v1+v2)/2)*(vel2-vel1)*mass[index2]*m_m/d2;
			
				// surface tension force
				double b = 32*M_PI*powf(r2,9);
				double m_d = -(945/b)*((r2*r2)-(d*d))*((3*r2*r2)-(7*d*d));
				fS += m_d*(m2/d2);

				// normale evaluation
				double mk = -945/b;
				N = N + P1P2*(m2/d2)*pow((r2*r2)-(d*d),2)*mk;
				//}
			}
		}
		fP = fP*(-d1);
		double lNi = length(N);
		double3 fSurf;
		if(lNi>=l[index1]){
			fSurf = fS*((-surfaceTension[index1])*N/lNi);
			N = normalize(N);
		}
		else {
			N = make_double3(0,0,0);
			fSurf = make_double3(0,0,0);
		}
		forcePressure[index1] = fP;
		forceViscosity[index1] = fV;
		normales[index1] = N;
		forceSurface[index1] = fSurf;
		forcesAccum[index1] = fP + fV + fSurf;
	}
}

/******************************************************************************************************************/
/******************************************************************************************************************/
__global__ void integrateSPH_LeapFrog_Forces(double3* velInterAv, double3* velInterAp, double3* oldPos, double3* newPos, 
				      double3* forcesViscosity, double3* forcesSurface, double3* forcesPressure, double3* forcesAccum,
				      double*  densities, float dt, uint nbBodies)
{
   uint index =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   if(index<nbBodies){
		double d = densities[index];
		if(d>0){
			// LEAP-FROG SCHEME (see [Kelager06])
			double3 F;
			F.x = (forcesAccum[index].x + forcesViscosity[index].x + forcesPressure[index].x + forcesSurface[index].x)/densities[index];
			F.y = (forcesAccum[index].y + forcesViscosity[index].y + forcesPressure[index].y + forcesSurface[index].y)/densities[index];
			F.z = (forcesAccum[index].z + forcesViscosity[index].z + forcesPressure[index].z + forcesSurface[index].z)/densities[index];
			velInterAv[index] = velInterAp[index];
			velInterAp[index] = velInterAp[index] + F*dt; 
			oldPos[index] = newPos[index];
			newPos[index] = newPos[index] + velInterAp[index]*dt;
		}
   }
}
/******************************************************************************************************************/
/******************************************************************************************************************/
__global__ void integrateSPH_LeapFrog(double3* velInterAv, double3* velInterAp, double3* oldPos, double3* newPos, 
				      double3* forces, double*  densities, float dt, uint nbBodies)
{
   uint index =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   if(index<nbBodies){
		double d = densities[index];
		if(d>0 && length(forces[index])>0){
		// LEAP-FROG SCHEME (see [Kelager06])
		/*velInterAv[index] = velInterAp[index];
		velInterAp[index] = velInterAp[index] + (forces[index]/d)*dt; 
		oldPos[index] = newPos[index];
		newPos[index] = newPos[index] + velInterAp[index]*dt;
*/
		velInterAv[index] = velInterAp[index];
		velInterAp[index] = velInterAp[index] + (forces[index]/d)*dt; 
		
		oldPos[index] = newPos[index];
		newPos[index] = newPos[index] + velInterAp[index]*dt;
		}

   }
}

/******************************************************************************************************************/
/******************************************************************************************************************/
__global__ void interpolateSPH_velocities(double3* velInterAv, double3* velInterAp, double3* oldVel, double3* newVel, uint nbBodies)
{
   uint index =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   if(index<nbBodies){
	//oldVel[index] = newVel[index];
	//newVel[index] = (velInterAv[index] + velInterAp[index])/2;
	oldVel[index] = newVel[index];
	newVel[index] = (velInterAv[index] + velInterAp[index])/2;
   }
}
/******************************************************************************************************************/
/******************************************************************************************************************/
__global__ void postProcessCollide_Kernel(double3* velInterAv, double3* velInterAp, double3* oldVel, double3* newVel, uint nbBodies)
{
   uint index =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   if(index<nbBodies){
	velInterAv[index] = oldVel[index];
	velInterAp[index] = newVel[index];
   }
}
/******************************************************************************************************************/
/******************************************************************************************************************/
}

