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
__global__ void compute_Pressure_PCI(uint nbBodies, double3* pos, double* mass, double* radius, double* k, double threshold, 
				     double* density, double* restDensity, double* densityError, 
			             double* pressure, partVoisine voisines, float dt)
{
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
		if((density[index]>restDensity[index])){
			double beta = 2*pow(mass[index]*dt/restDensity[index],2);
			double press = density[index]-restDensity[index];
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
					W2_ij += dot(dW,dW);
					W1_ij = W1_ij + dW;
				}
			}
			if(W2_ij>0){
				press = -press/(beta*(-dot(W1_ij,W1_ij) - W2_ij));
				pressure[index] += press;
			}
		}
      }
}			
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Density_PCI(uint nbBodies, double3* pos, double* mass, double* radius, 
			            double* density, double* restDensity, double* densityError, partVoisine voisines)
{
	uint index1 =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if(index1<nbBodies){
		density[index1] = 0;
		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){
			int index2 = voisines.listeVoisine[(index1*200)+i];			
			double h = max(radius[index1],radius[index2]);
			double d = length(pos[index1] - pos[index2]);
			density[index1] += (315/(64*M_PI*pow(h,9)))*pow((h*h)-(d*d),3)*mass[index2];
		}
		densityError[index1] = fabs(density[index1] - restDensity[index1])/100;
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Pressure_Force_PCI(uint nbBodies, double3* pos, double* mass, 
					   double* density, double* restDensity, double* densityError, 
					   double* pressure, double* radius, double3* forcesPressure, 
				           double threshold, partVoisine voisines)
{
	int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index < nbBodies)
	{
			forcesPressure[index] = make_double3(0,0,0);
			double pV1 = pressure[index]/(density[index]*density[index]);
			for(unsigned int i=0;i<voisines.nbVoisines[index];i++){
				int indexV = voisines.listeVoisine[(index*200)+i];
				double3 P1P2 = pos[index]-pos[indexV];
				double h = max(radius[index],radius[indexV]);
				double d = length(pos[index]-pos[indexV]);
				if(d>0){
					// Pression force
					double3 P1P2_N = P1P2/d;
					double pV2 = pressure[indexV]/(density[indexV]*density[indexV]);
					double a = M_PI*pow(h,6);
					double3 m_k = P1P2_N*(-45*pow(h-d,2)/a);			
					forcesPressure[index] = forcesPressure[index] + m_k*(pV1+pV2)*mass[indexV];
				}
			}
			forcesPressure[index] = forcesPressure[index]*(-density[index]);		
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Viscosity_SurfaceTension_Force_PCI(uint nbBodies, double3* pos, double3* vel, double* mass, double* density,  
			       			      double* pressure, double* radius, double* viscosity, double* l, 
						      double* surfaceTension, double3* normales,
			       			      double3* forcePressure, double3* forceViscosity, double3* forceSurface, 
						      double3* forcesAccum, partVoisine voisines)
{

        uint index1 =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
        if(index1<nbBodies){
		forcePressure[index1] = make_double3(0,0,0);
		forceViscosity[index1] = make_double3(0,0,0);
		forceSurface[index1] = make_double3(0,0,0);
		normales[index1] = make_double3(0,0,0);
		pressure[index1] = 0;
		double fSurface = 0;

		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){
			int index2 = voisines.listeVoisine[(index1*200)+i];
			double3 P1P2 = pos[index1] - pos[index2];
			double d = length(P1P2);
			double h = max(radius[index1],radius[index2]);
			if(d>0)
			{
                		// Viscosity force
				double a = M_PI*pow(h,6);
				double  m_m = 45*(h-d)/a;
				forceViscosity[index1] = forceViscosity[index1] + ((viscosity[index1]+viscosity[index2])/2)
							*(vel[index2]-vel[index1])*mass[index2]
							*m_m/density[index2];
				
				// surface tension force
				double b = 32*M_PI*powf(h,9);
				double m_d = -(945/b)*((h*h)-(d*d))*((3*h*h)-(7*d*d));
				fSurface += m_d*(mass[index2]/density[index2]);

				// normale evaluation
				double aN = 32*M_PI*pow(h,9);
				double mk = -945/aN;
				normales[index1] = normales[index1] + P1P2*(mass[index2]/density[index2])*pow((h*h)-(d*d),2)*mk;
			}
		}
		double lNi = length(normales[index1]);
		if(lNi>=l[index1]){
			surfaceTension[index1] = 0.0728;
			forceSurface[index1] = fSurface*(-surfaceTension[index1])*normales[index1]/lNi;
			normales[index1] = normalize(normales[index1]);
		}
		else {
			normales[index1] = make_double3(0,0,0);
		}
		forcesAccum[index1] = make_double3(0,-9.81*density[index1],0);//forcesAccum[index1]*density[index1];
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void calculate_Dt_CFL(uint nbBodies, double* radius, double* viscosities,
			         double3* forcesViscosity, double3* forcesSurface, double3* forcesPressure, double3* forcesAccum,
				 double*  densities, float *dt)
{
	// Calcul à revoir
	uint index =  blockIdx.x* blockDim.x + threadIdx.x;
	if(index < nbBodies){
		double h = radius[index];
		double3 F = forcesAccum[index] + forcesViscosity[index] + forcesPressure[index] + forcesSurface[index];
		double3 a1 = make_double3(F.x/densities[index],F.y/densities[index],F.z/densities[index]);
		float dt1 = min(pow(h/length(a1),0.5),0.5*h*h/viscosities[index]);
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
   uint index =  blockIdx.x* blockDim.x + threadIdx.x;

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
