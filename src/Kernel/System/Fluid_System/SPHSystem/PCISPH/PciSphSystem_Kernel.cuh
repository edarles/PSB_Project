#ifndef _PCI_SPH_KERNEL_
#define _PCI_SPH_KERNEL_

#include <SphKernel.cuh>
#include <common.cuh>

extern "C" {

__global__ void compute_Density_Pressure_PCI(uint nbBodies, double3* pos, double* mass, double* radius, double* k, double threshold, 
					     double* m_density, double* restDensity, double* m_densityError, 
			                     double* pressure, partVoisine voisines);

__global__ void compute_Density_PCI(uint nbBodies, double3* pos, double* mass, double* radius, 
			            double* density, double* restDensity, double* densityError, double* k, double* pressure, partVoisine voisines);

__global__ void compute_Pressure_PCI(uint nbBodies, double3* pos, double* mass, double* radius, double* k, double threshold, 
				     double* density, double* restDensity, double* densityError, 
			             double* pressure, partVoisine voisines, float dt);

__global__ void compute_Pressure_Force_PCI(uint nbBodies, double3* pos, double* mass, double* density, 
				           double* densityError, double* pressure, double* radius, double3* forcesPressure, 
				           double threshold, uint nbIter, partVoisine voisines);


__global__ void compute_Viscosity_SurfaceTension_Force_PCI(uint nbBodies, double3* pos, double3* vel, double* mass, double* density,  
			       			      double* radius, double* viscosity, double* l, double* surfaceTension, double3* normales,
			       			      double3* forceViscosity, double3* forceSurface, double3* forcesAccum, partVoisine voisines);

__global__ void calculate_Dt_CFL(uint nbBodies, double* radius, double* viscosities,
			         double3* forcesViscosity, double3* forcesSurface, double3* forcesPressure, double3* forcesAccum,
				 double*  densities, float dt);

__global__ void integrate_SPH_LeapFrog_PCI(double3* velInterAv, double3* velInterAp, double3* oldPos, double3* newPos, 
				      double3* forcesViscosity, double3* forcesSurface, double3* forcesPressure, double3* forcesAccum,
				      double*  densities, float dt, uint nbBodies);


__global__ void restore_Pos_Vel_PCI(double3* restore_velInterAv, double3* restore_velInterAp, double3* restore_oldPos, 
			            double3* restore_newPos, double3* velInterAv, double3* velInterAp, double3* oldPos, 
				    double3* newPos, uint nbBodies);

__global__ void  evaluateChanges_T_Visc_Rho0_mass_Kernel(double3* pos, double* m_dMass, double* m_interactionRadius, 
				       double* m_density, double* m_temperatures, double* m_viscosity, partVoisine voisines,
				       double* m_Dtemperatures, double* m_Dviscosity, double *m_Dmass,uint nbParticles);

__global__ void  integrateChanges_T_Visc_Rho0_mass_Kernel(double3* pos, double* radius, double* kernelParticles, double* m_temperatures, 
				       double* m_viscosity, double* m_restDensity,  double *m_mass, double* m_Dtemperatures, double* m_Dviscosity, 
				       double* m_DrestDensity, double *m_Dmass, 
				       partVoisine voisines, uint nbParticles);

}

#endif