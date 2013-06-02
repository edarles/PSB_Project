#include <MSphSystem_Kernel.cuh>
#include <double3.h>
#include <stdio.h>

extern "C"
{
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void evaluate_diffusion_coefficient_Kernel(double3* pos, double* mass, double* radius, 
				    	       double* density, double* ci, double* temp, double* visc, partVoisine voisines,
					       double* cij, uint nbBodies)
{
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	double3 pos1 = pos[index];
	double r1 = radius[index];
	//double T1 = temp[index];
	double mu1 = visc[index];
	double c1 = ci[index];
	double delta = 0;

	for(uint i=0;i<voisines.nbVoisines[index];i++){
		uint indexV = voisines.listeVoisine[index*200+i];
		double3 pos2 = pos[indexV];
		double r2 = radius[indexV];
		double h = max(r1,r2);
		double d = length(pos1 - pos2);
		double W = (315/(64*M_PI*pow(h,9)))*pow((h*h)-(d*d),3);
		if(fabs(visc[indexV]-mu1)>0)
			delta += (mass[indexV]/density[indexV])*c1*(1/fabs(visc[indexV]-mu1))*W;
		else    
			delta += (mass[indexV]/density[indexV])*c1*W;
	}
	cij[index] = delta;
	printf("cij:%f\n",delta);
    }
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void evaluate_T_Rho0_Mass_Kernel (double3* pos, double* mass, double* radius, double* density,
					     double* restDensity, double* temp, double* cij, partVoisine voisines,
				     	     double* deltaT, double* deltaRho0, double* deltaM,
					     uint nbBodies)
{
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	double3 pos1 = pos[index];
	double r1 = radius[index];
	double T1 = temp[index];
	double rho1 = restDensity[index];
	double m1 = mass[index];
	//double c1 = cij[index];
	double dTemp = 0;
	double dRho0 = 0;
	double dM = 0;
	double c = 10.0; // diffusion coefficient
	for(uint i=0;i<voisines.nbVoisines[index];i++){
		uint indexV = voisines.listeVoisine[index*200+i];
		double3 pos2 = pos[indexV];
		double r2 = radius[indexV];
		double h = max(r1,r2);
		double d = length(pos1 - pos2);
		if(d>0){
			double a = 64*M_PI*pow(h,9);
			double m_k = 315/a;
			double b = pow((h*h)-(d*d),3);
			double d2W = b*m_k;//(-945/(32*M_PI*pow(h,9)))*(h*h-d*d)*(3*h*h-7*d*d);//*cij[indexV];
			dTemp += c*(mass[indexV]/density[indexV])*(temp[indexV]-T1)*d2W;//*d2W;//*cij[indexV];
			dRho0 += c*(mass[indexV]/density[indexV])*(restDensity[indexV]-rho1)*d2W;//*cij[indexV];
			dM += c*(mass[indexV]/density[indexV])*(mass[indexV]-m1)*d2W;//*cij[indexV];
		}
	}
	deltaT[index] = dTemp;///(TempTotal*density[index]);
	deltaRho0[index] = dRho0;//voisines.nbVoisines[index]*dRho0*density[index];
	deltaM[index] = dM;//voisines.nbVoisines[index]*dM*density[index];
   }
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void integrate_T_Rho0_Mass_Kernel(double *T, double* restDensity, double* mass,
					     double TS, double MS, double Rho0S,
					     double* dT, double* dRho0, double *dM, double dt, uint nbBodies)
{
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	T[index] += dT[index]*dt;
	mass[index] += dM[index]*dt;
	restDensity[index] += dRho0[index]*dt;
	//if(dT[index]>0)printf("Temp:%f m:%f rho0:%f\n",T[index],mass[index],restDensity[index]);
    }
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
__global__ void evaluate_viscosity_Kernel(double* visc, double* dT, double R, double EA, double dt, uint nbBodies)
{
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	double deltaVisc = visc[index]*exp(EA/(R*dT[index]));
        visc[index] += deltaVisc*dt;
    }
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
