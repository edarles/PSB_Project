#include <MSphSystem_Kernel.cuh>
#include <double3.h>
#include <stdio.h>

extern "C"
{

/****************************************************************************************************************************/
// Diffusion coefficient evolution of a particle i of a phase k
/****************************************************************************************************************************/
__global__ void diffusionEvolution_Kernel(double* dD, double* D0, double* dT, double EA, double* R, int nbBodies, int nbPhases, int indexPhase)
{
	
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	double dTemp = dT[index*nbPhases+indexPhase];
	if(dTemp>0 && R>0)
		dD[index*nbPhases+indexPhase] = D0[index*nbPhases+indexPhase]*expf(-1.0/(R[index]*dTemp));
    }
}

/****************************************************************************************************************************/
// Temperature evolution of a particle i of a phase k
/****************************************************************************************************************************/
__global__ void temperatureEvolution_Kernel(double* dT, double* D, double* mass, double* T, double* densBarre, 
					    double3* pos, double* radius, partVoisine voisines, int* partPhases, int nbBodies, int nbPhases, int indexPhase)
{
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	double3 pos1 = pos[index];
	double r1 = radius[index];
	double Temp = T[index*nbPhases+indexPhase];
	double dTemp = 0;
	double Diff = D[index*nbPhases+indexPhase];

	for(uint i=0;i<voisines.nbVoisines[index];i++){

		uint indexV = voisines.listeVoisine[index*200+i];
		double3 pos2 = pos[indexV];
		double d = length(pos1 - pos2);

		if(d>0){ 
			double r2 = radius[indexV];
			double h = max(r1,r2);
			double d2 = densBarre[indexV];

			double a = 64*M_PI*pow(h,9);
			double m_k = 315/a;
			double b = pow((h*h)-(d*d),3);
			double d2W = b*m_k;

			dTemp = dTemp + ((D[index*nbPhases+indexPhase]+D[indexV*nbPhases+indexPhase])/2)*d2W*(mass[indexV]/d2)*(T[indexV*nbPhases+indexPhase]-Temp);
		}
	}
	dT[index*nbPhases+indexPhase] = dTemp;
	//if(nbPhases>1 && dTemp>0) printf("dTemp:%f\n",dTemp);
    }
}

/****************************************************************************************************************************/
// Viscosity evolution of a particle i of a phase k
// A revoir -> le faire par voisinage SPH pour diffusion de proche en proche
/****************************************************************************************************************************/
__global__ void viscosityEvolution2_Kernel(double* dMu, double* dT, double* T, double *dD, double* D, double* R, double* radius, double* densRest, double3* pos, 
					   double* mass, double* dens, partVoisine voisines,
					   int nbBodies, int nbPhases, int indexPhase)
{
	
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	double dDiff = dD[indexPhase*nbPhases+index];
	double dTemp = dT[indexPhase*nbPhases+index];
	double densRestK = densRest[indexPhase*nbPhases+index];
	double Temp = T[indexPhase*nbPhases+index];
	double Diff = D[indexPhase*nbPhases+index];
	double dmu = 0;
        double3 pos1 = pos[index];
        double r1 = radius[index];
	//printf("R:%f\n",R[index]);
	for(uint i=0;i<voisines.nbVoisines[index];i++){

		uint indexV = voisines.listeVoisine[index*200+i];
		double3 pos2 = pos[indexV];
		double d = length(pos1 - pos2);
		double dDiffV = dD[indexPhase*nbPhases+indexV];
		double dTempV = dT[indexPhase*nbPhases+indexV];
		double TempV = T[indexPhase*nbPhases+indexV];
		double DiffV = D[indexPhase*nbPhases+indexV];
		double densRestKV = densRest[indexPhase*nbPhases+indexV];
		double diff1 = (-dTemp*Diff+Temp*dDiff)/(Diff*Diff);
		double diff2 = (-dTempV*DiffV+TempV*dDiffV)/(DiffV*DiffV);


		if(d>0){ 
			double r2 = radius[indexV];
			double h = max(r1,r2);
			double d2 = dens[indexV];
			
			double a = 64*M_PI*pow(h,9);
			double m_k = 315/a;
			double b = pow((h*h)-(d*d),3);
			double d2W = b*m_k;

			dmu = dmu + d2W*(mass[indexV]/d2)*(diff2-diff1);
		}
	}
	dmu = -dmu*(R[index]/(6*M_PI*radius[index]));
	//if(dmu!=0)printf("i:%d dmu:%f\n",index,dmu);
	dMu[indexPhase*nbPhases+index] = dmu;//*mass[index];
    }
}
/****************************************************************************************************************************/
__global__ void viscosityEvolution_Kernel(double* dMu, double* dT, double* T, double *dD, double* D, double* R, double* radius, double* densRest, int nbBodies, int nbPhases, int indexPhase)
{
	
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	double dDiff = dD[index*nbPhases+indexPhase];
	double dTemp = dT[index*nbPhases+indexPhase];
	double densRestK = densRest[index*nbPhases+indexPhase];
	double Temp = T[index*nbPhases+indexPhase];
	double Diff = D[index*nbPhases+indexPhase];
	if(densRest>0 && dDiff!=0){
		dMu[index*nbPhases+indexPhase]=-(1/densRestK)*(R[index]/(6*M_PI*radius[index]))*(-dTemp*Diff+Temp*dDiff)/(Diff*Diff);
	}
    }
}
/****************************************************************************************************************************/
// Integrate Diffusion, Temperature and viscosity evolution
/****************************************************************************************************************************/
__global__ void integrate_D_T_Mu_Kernel(double* D, double *Dk, double *T, double* Tk, double *Mu, double *dD, double *dT, 
					double *dMu, double *alpha, double TMin, double TMax, double muMin, double muMax,
					double dt, int nbBodies, int nbPhases, int maxNumberPhases)
{
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	double dMu_i = 0;
	double dD_i = 0;
	double dT_i = 0;
	for(int i=0; i<nbPhases; i++)
	{
		//printf("alpha:%f\n",alpha[i*maxNumberPhases+index]);
		dMu_i = dMu_i + dMu[index*maxNumberPhases+i]*(alpha[index*maxNumberPhases+i]);
		dD_i = dD_i + dD[index*maxNumberPhases+i]*(alpha[index*maxNumberPhases+i]);
		dT_i = dT_i + dT[index*maxNumberPhases+i]*alpha[index*maxNumberPhases+i];
		Dk[index*maxNumberPhases+i] = Dk[index*maxNumberPhases+i] + dD[index*maxNumberPhases+i]*alpha[index*maxNumberPhases+i]*dt;
		Tk[index*maxNumberPhases+i] = Tk[index*maxNumberPhases+i] + dT[index*maxNumberPhases+i]*alpha[index*maxNumberPhases+i]*dt;
		//if(index==0) printf("Dk:%f\n",Dk[i*nbPhases+index]);
	}
	D[index]=D[index]+dD_i*dt;
	T[index]=T[index]+dT_i*dt;
	Mu[index]=Mu[index]+dMu_i*dt;
	
	if(index==20) printf("D:%f T:%f Mu:%f\n",D[index],T[index],Mu[index]);

	if(T[index]>TMax) T[index]=TMax;
	if(T[index]<TMin) T[index]=TMin;

	if(Mu[index]>muMax) Mu[index]=muMax;
	if(Mu[index]<muMin) Mu[index]=muMin;

	//if(dMu_i!=0) printf("i:%d Mu:%f\n",index, Mu[index]);
	
    }
}
/****************************************************************************************************************************/
// Concentration evolution with second Fick's law
/****************************************************************************************************************************/
__global__ void evolutionConcentration_Kernel(double* dAlphak, double* alphak, double *Dk, double3* pos, double* mass, double* dens, double* radius, int* partPhases,
					      partVoisine voisines, int nbBodies, int nbPhases, int indexPhase)
{
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
	double alpha1 = alphak[index*nbPhases + indexPhase];
	double3 pos1 = pos[index];
	double r1 = radius[index];
	double m1 = mass[index];
	double dens1 = dens[index];
	double D1 = Dk[index*nbPhases + indexPhase];

	double laplC = 0;
	double3 gradC = make_double3(0,0,0);
	double3 gradD = make_double3(0,0,0);

        for(uint i=0;i<voisines.nbVoisines[index];i++){
		uint indexV = voisines.listeVoisine[index*200+i];

		if(partPhases[indexV]!=partPhases[index]){

		double alpha2 = alphak[indexV*nbPhases + indexPhase];
		double3 pos2 = pos[indexV];
		double r2 = fmax(r1,radius[indexV]);
		double m2 = mass[indexV];
		double dens2 = dens[indexV];
		double D2 = Dk[indexV*nbPhases + indexPhase];
		double d = length(pos1 - pos2);

		if(d>0){
			double a = M_PI*pow(r2,6);
			double3 P1P2_N = (pos1-pos2)/d;
			double3 m_k = P1P2_N*(-45*pow(r2-d,2)/a);
			gradC = gradC + m2*((alpha1/(dens1*dens1))+(alpha2/(dens2*dens2)))*m_k;
			gradD = gradD + m2*((D1/(dens1*dens1))+(D2/(dens2*dens2)))*m_k;
			//double  m_m = 45*(r2-d)/a;
			laplC = laplC + (m2/dens2)*((alpha1-alpha2))*dot((m_k/(0.01*r2*r2)),P1P2_N);
		}
		}
	}
	double result = 0;
	gradC = dens1*gradC;
	gradD = dens1*gradD;
	laplC = 2*laplC;
	if(laplC!=0) {
		result = (laplC*D1 + dot(gradC,gradD))/dens1;
		//printf("result:%f\n",result);
	}
	dAlphak[index*nbPhases+indexPhase] = result;
     }
}
/****************************************************************************************************************************/
__global__ void integrateConcentration_Kernel(double* dAlphak, double* alphak, double dt, int nbBodies, int nbPhases, int maxNumberPhases)
{
    int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if(index < nbBodies)
    {
        for(int i=0;i<nbPhases;i++){
		double alphakt = alphak[index*maxNumberPhases + i] + dAlphak[index*maxNumberPhases + i]*dt;
		if(alphakt<0) alphakt = 0;
		if(alphakt>1) alphakt = 1;//alphakt = fmax(fmin(alphakt,1),0);
		alphak[index*maxNumberPhases + i] = alphakt;
		//if(i==0) printf("alpha:%f\n",alphakt);
	}
    }
}
/****************************************************************************************************************************/
}
