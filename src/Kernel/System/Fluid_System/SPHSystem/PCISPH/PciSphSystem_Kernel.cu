#include <PciSphSystem_Kernel.cuh>
#include <double3.h>
#include <stdio.h>
#include <AtomicDoubleAdd.cuh>

extern "C"
{
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__device__ float fatomicMin(float *addr, float value)
{
        float old = *addr, assumed;
        if(old <= value) return old;
        do {
                assumed = old;
                old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

        }while(old!=assumed);
        return old;
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__device__ float fatomicMax(float *addr, float value)
{
        float old = *addr, assumed;
        if(old >= value) return old;
        do {
                assumed = old;
                old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

        }while(old!=assumed);
        return old;
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Pressure_PCI(uint nbBodies, double3* pos, double* mass, double* radius, double threshold, 
				     double* density, double* restDensity, double* densityError, double* k,
			             double* pressure, partVoisine voisines, float dt)
{
    int index =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
			double beta = 2*mass[index]*mass[index]*dt*dt/(restDensity[index]*restDensity[index]);
			double3 W1_ij = make_double3(0,0,0);
			double W2_ij = 0;
			for(unsigned int i=0;i<voisines.nbVoisines[index];i++){
				int indexV = voisines.listeVoisine[index*200+i];
				double3 P1P2 = pos[index] - pos[indexV];
				double h = max(radius[index],radius[indexV]);
				double d = length(P1P2);
				if(d>0){
					double3 P1P2_N = P1P2/d;
					double a = M_PI*pow(h,6);
					double3 dW = P1P2_N*(-45*pow(h-d,2)/a);
					W2_ij += (dW.x*dW.x)+(dW.y*dW.y)+(dW.z*dW.z);
					W1_ij = W1_ij + dW;
				}
			}
			double dotW = W1_ij.x*W1_ij.x + W1_ij.y*W1_ij.y + W1_ij.z*W1_ij.z;
			double press1 = -densityError[index]/(beta*(-dotW - W2_ij));
			double press2 = k[index]*restDensity[index]*(powf(density[index]/restDensity[index],7)-1);//k[index]*densityError[index];//k[index]*restDensity[index]*(powf(density[index]/restDensity[index],7)-1);
			if(densityError[index]<0)
				pressure[index] += press2;
			else
				pressure[index] += press1;
      }
}			
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Density_PCI(uint nbBodies, double3* pos, double* mass, double* radius, 
			            double* density, double* restDensity, double* densityError, double* pressure, partVoisine voisines)
{
	uint index1 =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if(index1<nbBodies){
		density[index1] = 0;
		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){
			int index2 = voisines.listeVoisine[(index1*200)+i];			
			double h = max(radius[index1],radius[index2]);
			double d = length(pos[index1] - pos[index2]);
			density[index1] += (315/(64*M_PI*pow(h,9)))*pow((h*h)-(d*d),3)*mass[index2];
		}
		densityError[index1] = density[index1] - restDensity[index1];
		pressure[index1] = 0;
	}
}
/******************************************************************************************************************************************/
__global__ void compute_Density_PCI_predict(uint nbBodies, double3* pos, double* mass, double* radius, 
			            double* density, double* restDensity, double* densityError, partVoisine voisines)
{
	uint index1 =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if(index1<nbBodies){
		density[index1] = 0;
		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){
			int index2 = voisines.listeVoisine[(index1*200)+i];			
			double h = max(radius[index1],radius[index2]);
			double d = length(pos[index1] - pos[index2]);
			density[index1] += (315/(64*M_PI*pow(h,9)))*pow((h*h)-(d*d),3)*mass[index2];
		}
		densityError[index1] = density[index1] - restDensity[index1];
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Pressure_Force_PCI(uint nbBodies, double3* pos, double* mass, 
					   double* density, double* restDensity, double* densityError, 
					   double* pressure, double* radius, double3* forcesPressure, double threshold, partVoisine voisines)
{
	int index =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index < nbBodies)
	{
			if(fabs(densityError[index])>10){
			double3 fP = make_double3(0,0,0);
			double pV1 = pressure[index]/(density[index]*density[index]);

			for(unsigned int i=0;i<voisines.nbVoisines[index];i++){

				int indexV = voisines.listeVoisine[(index*200)+i];
				double3 P1P2 = pos[index]-pos[indexV];
				double h = max(radius[index],radius[indexV]);
				double d = length(pos[index]-pos[indexV]);

				if(d>0){
					// Pression force
					double pV2 = pressure[indexV]/(density[indexV]*density[indexV]);
					double3 P1P2_N = P1P2/d;
					double a = M_PI*pow(h,6);
					double3 m_k = P1P2_N*(-45*pow(h-d,2)/a);			
					fP =  fP + m_k*(pV1+pV2)*mass[indexV];
				}
			}
			forcesPressure[index] = fP*(-density[index]);	
			}	
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Viscosity_SurfaceTension_Force_PCI(uint nbBodies, double3* pos, double3* vel, double* mass, double* density,  
			       			      double* pressure, double* radius, double* viscosity, double* l, 
						      double* surfaceTension, double3* normales,
			       			      double3* forceViscosity, double3* forceSurface, double3* forcePressure, partVoisine voisines)
{

           uint index1 =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
        if(index1<nbBodies){

		register double3 fV = make_double3(0,0,0);
		register double3 N = make_double3(0,0,0);
		register double fS = 0;

		// attributes declaration;
		double v1 = viscosity[index1];
		double3 vel1 = vel[index1];
		double r1 = radius[index1];

		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){

			int index2 = voisines.listeVoisine[(index1*200)+i];
			double3 P1P2 = pos[index1] - pos[index2];
			double d = length(P1P2);

			if(d>0)
			{
				// attributes declaration
				double d2 = density[index2];
				double r2 = max(radius[index2],r1);
				double m2 = mass[index2];
				double v2 = viscosity[index2];
				double3 vel2 = vel[index2];
	
                		// Viscosity force
				double a = M_PI*pow(r2,6);
				double  m_m = 45*(r2-d)/a;
				fV = fV + ((v1+v2)/2)*(vel2-vel1)*mass[index2]*m_m/d2;
			
				// surface tension force
				double b = 32*M_PI*powf(r2,9);
				double m_d = -(945/b)*((r2*r2)-(d*d))*((3*r2*r2)-(7*d*d));
				fS += m_d*(m2/d2);

				// normale evaluation
				double mk = -945/b;
				N = N + P1P2*(m2/d2)*pow((r2*r2)-(d*d),2)*mk;
			}
		}
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
		forcePressure[index1] = make_double3(0,0,0);
		forceViscosity[index1] = fV;
		normales[index1] = N;
		forceSurface[index1] = fSurf;
		//forcesAccum[index1] = fP + fV + fSurf;
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void calculate_Dt_CFL(uint nbBodies, double* radius, double* viscosities,
			         double3* forcesViscosity, double3* forcesSurface, double3* forcesPressure, double3* forcesAccum,
				 double*  densities, float *dt)
{
	// Calcul à revoir
	uint index =  __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index < nbBodies){
		double h = radius[index];
		double3 F = forcesAccum[index] + forcesViscosity[index] + forcesPressure[index] + forcesSurface[index];
		double3 a1 = make_double3(F.x/densities[index],F.y/densities[index],F.z/densities[index]);
		float dt1 = fmin(pow(h/length(a1),0.5),0.5*h*h/viscosities[index]);
		fatomicMin(dt,dt1);
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void restore_Pos_Vel_PCI(double3* restore_velInterAv, double3* restore_velInterAp, double3* restore_oldPos, 
			            double3* restore_newPos, double3* velInterAv, double3* velInterAp, 
				    double3* oldPos, double3* newPos, uint nbBodies)
{
   uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

   if(index<nbBodies){
		velInterAv[index] = restore_velInterAv[index];
		velInterAp[index] = restore_velInterAp[index];
		oldPos[index] = restore_oldPos[index];	
		newPos[index] = restore_newPos[index];
   }
}	
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
}
