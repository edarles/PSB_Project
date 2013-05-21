#include <PciSphSystem_Kernel.cuh>
#include <double3.h>
#include <AtomicDoubleAdd.cuh>

extern "C"
{
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Density_Pressure_PCI(uint nbBodies, double3* pos, double* mass, double* radius, double* k, double threshold, 
					     double* density, double* restDensity, double* densityError, 
			                     double* pressure, partVoisine voisines)
{
	int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index < nbBodies)
	{
		if(density[index]>restDensity[index]){
			double3 pos1 = pos[index];
			double  h1 = radius[index];
			for(unsigned int i=0;i<voisines.nbVoisines[index];i++){
				int indexV = voisines.listeVoisine[index*200+i];
				double3 pos2 = pos[indexV];
				double3 P1P2 = pos1 - pos2;
				double d = length(P1P2);
				double r = max(h1,radius[indexV]);
				double a = 64*M_PI*pow(r,9);
				double m_k = 315/a;
				double b = pow((r*r)-(d*d),3);
				density[index] += b*m_k*mass[indexV];
			}
			densityError[index] = fabs(density[index] - restDensity[index]);
			double omega = 7.0;
			pressure[index] = k[index]*(restDensity[index]/omega)*(pow(density[index]/restDensity[index],omega)-1);
		}
	}
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
		if(density[index]>restDensity[index]){
			double beta = 2*pow(mass[index]*dt/restDensity[index],2);
			double3 pos1 = pos[index];
			double  h1 = radius[index];
			double press = (restDensity[index]-density[index]);
			double3 W1_ij = make_double3(0,0,0);
			double W2_ij = 0;
			for(unsigned int i=0;i<voisines.nbVoisines[index];i++){
				int indexV = voisines.listeVoisine[index*200+i];
				double3 pos2 = pos[indexV];
				double3 P1P2 = pos1 - pos2;
				double h = max(h1,radius[indexV]);
				double d = length(P1P2);
				/*double3 P1P2_N = normalize(P1P2);
				double3 m_k = make_double3(P1P2_N.x,P1P2_N.y,P1P2_N.z);
				double a = M_PI*pow(h,6);
				double3 dW = m_k*(-45*pow(h-d,2)/a);
				*/
				double3 dW = -945/(32*M_PI*pow(h,9))*P1P2*pow(h*h-d*d,2);
				W2_ij += dot(dW,dW);
				W1_ij = W1_ij + dW;
			}
			press = press/(beta*(-dot(W1_ij,W1_ij) - W2_ij));
			pressure[index] += press;
		}
      }
}			
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Density_PCI(uint nbBodies, double3* pos, double* mass, double* radius, 
			            double* density, double* restDensity, double* densityError, double* k, double* pressure, partVoisine voisines)
{
	uint index1 =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if(index1<nbBodies){

		double3 pos1 = pos[index1];
		double  h1 = radius[index1];
		density[index1] = 0;
		
		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){
			int index2 = voisines.listeVoisine[(index1*200)+i];			
			double3 pos2 = pos[index2];
			double  mass2 = mass[index2];
			double  h2 = radius[index2];
			double h = max(h1,h2);
			double3 P1P2 = pos1 - pos2;
			double d = length(P1P2);
		        
			double a = 64*M_PI*pow(h,9);
			double m_k = 315/a;
			double b = pow((h*h)-(d*d),3);
			density[index1] += b*m_k*mass2;
		}
		double omega = 7;
		pressure[index1] = k[index1]*(restDensity[index1]/omega)*(pow(density[index1]/restDensity[index1],omega)-1);
		densityError[index1] = fabs(density[index1] - restDensity[index1]);
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Pressure_Force_PCI(uint nbBodies, double3* pos, double* mass, double* density, 
				           double* densityError, double* pressure, double* radius, double3* forcesPressure, 
				           double threshold, uint nbIter, partVoisine voisines)
{
	int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index < nbBodies)
	{
		forcesPressure[index] = make_double3(0,0,0);
			double3 pos1 = pos[index];
			double h1 = radius[index];
			double pV1 = pressure[index]/(density[index]*density[index]);
			for(unsigned int i=0;i<voisines.nbVoisines[index];i++){
				int indexV = voisines.listeVoisine[(index*200)+i];
				double3 pos2 = pos[indexV];
				double3 P1P2 = pos1 - pos2;
				double h2 = radius[indexV];
				double h = max(h1,h2);
				double d = length(P1P2);
				if(d>0){
					// Pression force
					double pV2 = pressure[indexV]/(density[indexV]*density[indexV]);
					double3 P1P2_N = normalize(P1P2);
					double3 m_k = make_double3(P1P2_N.x,P1P2_N.y,P1P2_N.z);
					double a = M_PI*pow(h,6);
					m_k = m_k*(-45*pow(h-d,2)/a)*(pV1+pV2);
				        //double3 m_k = -945/(32*M_PI*pow(h,9))*P1P2*pow(h*h-d*d,2)*(pV1+pV2);
					forcesPressure[index] = forcesPressure[index] + m_k;
				}
			}
			forcesPressure[index] = forcesPressure[index]*(-mass[index]);
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void compute_Viscosity_SurfaceTension_Force_PCI(uint nbBodies, double3* pos, double3* vel, double* mass, double* density,  
			       			      double* radius, double* viscosity, double* l, double* surfaceTension, double3* normales,
			       			      double3* forceViscosity, double3* forceSurface, double3* forcesAccum, partVoisine voisines)
{

        uint index1 =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
        if(index1<nbBodies){

		double3 pos1 = pos[index1];
		double3 vel1 = vel[index1];
		double h1 = radius[index1];
		double visc1 = viscosity[index1];

		forceViscosity[index1] = make_double3(0,0,0);
		forceSurface[index1] = make_double3(0,0,0);
		normales[index1] = make_double3(0,0,0);
		

		for(unsigned int i=0;i<voisines.nbVoisines[index1];i++){

			int index2 = voisines.listeVoisine[(index1*200)+i];
			double3 pos2 = pos[index2];
			double3 vel2 = vel[index2];
			double  dens2 = density[index2];
			double  mass2 = mass[index2];
			double  h2 = radius[index2];
			double  visc2 = viscosity[index2];

			double3 P1P2 = pos1 - pos2;
			double d = length(P1P2);
			double h = max(h1,h2);

			if(d>0)
			{
                		// Viscosity force
				double a = M_PI*pow(h,6);
				double  m_m = 45*(h-d)/a;
				forceViscosity[index1] = forceViscosity[index1] + ((visc1+visc2)/2)*(vel2-vel1)*mass2*m_m/dens2;
				
				// surface tension force
				double b = 32*M_PI*powf(h,9);
				double m_d = -(945/b)*((h*h)-(d*d))*((3*h*h)-(7*d*d));
				forceSurface[index1].x += m_d*(mass2/dens2);
				forceSurface[index1].y += m_d*(mass2/dens2);
				forceSurface[index1].z += m_d*(mass2/dens2);

				// normale evaluation
				double aN = 32*M_PI*pow(h,9);
				double mk = -945/aN;
				normales[index1] = normales[index1] + P1P2*(mass2/dens2)*pow((h*h)-(d*d),2)*mk;
			}
		}
		if(length(normales[index1])>0)
			normales[index1] = normalize(normales[index1]);
		
		if(length(normales[index1])>=l[index1]){
			forceSurface[index1].x *= (-surfaceTension[index1])*normales[index1].x;
			forceSurface[index1].y *= (-surfaceTension[index1])*normales[index1].y;
			forceSurface[index1].z *= (-surfaceTension[index1])*normales[index1].z;
		}
		else
			forceSurface[index1] = make_double3(0,0,0);

		forcesAccum[index1] = make_double3(0,-9.81*density[index1],0);
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void calculate_Dt_CFL(uint nbBodies, double* radius, double* viscosities,
			         double3* forcesViscosity, double3* forcesSurface, double3* forcesPressure, double3* forcesAccum,
				 double*  densities, float dt)
{
        for(unsigned int index=0; index<nbBodies; index++){
		double h = radius[index];
		double3 F = forcesAccum[index] + forcesViscosity[index] + forcesSurface[index] + forcesPressure[index];
		double3 a1 = make_double3(F.x/densities[index],F.y/densities[index],F.z/densities[index]);
		float dt1 = min(pow(h/length(a1),0.5),0.5*h*h/viscosities[index]);
		dt = min(dt,dt1);
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void integrate_SPH_LeapFrog_PCI(double3* velInterAv, double3* velInterAp, double3* oldPos, double3* newPos, 
				      double3* forcesViscosity, double3* forcesSurface, double3* forcesPressure, double3* forcesAccum,
				      double*  densities, float dt, uint nbBodies)
{
   uint index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

   if(index<nbBodies){
		// LEAP-FROG SCHEME (see [Kelager06])
		// x(t): position at time t
		double3 pos = newPos[index];
		// velInterAp is velocity at time t-0.5
		double3 vel = velInterAp[index];
		// a(t+1) = F(t+1)/d(t+1)
		double3 F = forcesAccum[index] + forcesViscosity[index] + forcesSurface[index] + forcesPressure[index];
		double3 a1 = make_double3(F.x/densities[index],F.y/densities[index],F.z/densities[index]);

		// store vel at time (t-0.5)
		velInterAv[index] = make_double3(vel.x,vel.y,vel.z);

		// v(t+0.5) = v(t-0.5) + a(t)*dt
		vel.x += a1.x*dt;
		vel.y += a1.y*dt;
		vel.z += a1.z*dt;
		velInterAp[index] = make_double3(vel.x,vel.y,vel.z);

		oldPos[index] = make_double3(newPos[index].x,newPos[index].y,newPos[index].z);
		// x(t+1) = x(t) + v(t+0.5)*dt
		pos.x += vel.x * dt;
		pos.y += vel.y * dt;
		pos.z += vel.z * dt;
		newPos[index] = make_double3(pos.x,pos.y,pos.z);

   }
}

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
__global__ void  evaluateChanges_T_Visc_Rho0_mass_Kernel(double3* pos, double* m_dMass, double* radius, 
				       double* m_density, double* m_temperatures, double* m_viscosity, partVoisine voisines,
				       double* m_Dtemperatures, double* m_Dviscosity, double *m_Dmass, uint nbParticles)
{
	int index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index < nbParticles)
	{
		m_Dtemperatures[index] = 0;
		m_Dviscosity[index] = 0;
		m_Dmass[index] = 0;
		double3 pos1 = pos[index];
		double r = radius[index];
		double a = 315/(100*M_PI*pow(r,6));
		
		for(unsigned int i=0;i<voisines.nbVoisines[index];i++){
			int indexV = voisines.listeVoisine[index*200+i];
			double3 pos2 = pos[indexV];
			double3 P1P2 = pos1 - pos2;
			double d = length(P1P2);
			double m_k = a*powf(((r*r)-(d*d)),3);
			m_Dtemperatures[index] += (m_dMass[indexV]/m_density[indexV])*((m_temperatures[indexV]-m_temperatures[index]))*m_k;
			m_Dviscosity[index] += (m_dMass[indexV]/m_density[indexV])*((m_viscosity[indexV]-m_viscosity[index]))*m_k;
			m_Dmass[index] += (m_dMass[indexV]/m_density[indexV])*((m_dMass[indexV]-m_dMass[index]))*m_k;
					//  (1-fabs((m_viscosity[indexV]-m_viscosity[index])/(m_viscosity[indexV]+m_viscosity[index])))*m_k;
			
			
		}
		double D = 0.5;
		m_Dtemperatures[index] = m_Dtemperatures[index]*m_density[index]/D;
		m_Dviscosity[index] = fabs(m_Dtemperatures[index])*m_Dviscosity[index]*m_density[index]/D;
		m_Dmass[index] = m_Dmass[index]*m_density[index]/D;
		
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
__global__ void  integrateChanges_T_Visc_Rho0_mass_Kernel(double3* pos, double* radius, double* kernelParticles, double* m_temperatures, 
				       double* m_viscosity, double* m_restDensity, double *m_mass, double* m_Dtemperatures, double* m_Dviscosity, 
				       double* m_DrestDensity, double *m_Dmass, 
				       partVoisine voisines,uint nbParticles)
{
	uint index =  __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index < nbParticles)
	{
		double3 pos1 = pos[index];
		double r = radius[index];
		m_DrestDensity[index] = 0;
		double a = 315/(100*M_PI*pow(r,6));
		for(unsigned int i=0;i<voisines.nbVoisines[index];i++){
			uint indexV = voisines.listeVoisine[index*200+i];
			double3 pos2 = pos[indexV];
			double3 P1P2 = pos1 - pos2;
			double d = length(P1P2);
			double m_k = a*powf(((r*r)-(d*d)),3);
			m_DrestDensity[index] += m_Dmass[indexV]*m_k;
		}
		double dt = 0.1;
		m_mass[index] += m_Dmass[index]*dt;
		m_restDensity[index] += m_DrestDensity[index]*dt;
		radius[index] = powf((3*m_mass[index]*kernelParticles[index])/(4*M_PI*m_restDensity[index]),0.333);		
		if(m_Dtemperatures[index]!=0){
			m_temperatures[index] += m_Dtemperatures[index]*dt;
			m_viscosity[index] += m_Dviscosity[index]*dt;
		}
	}
}
/******************************************************************************************************************************************/
/******************************************************************************************************************************************/
}
