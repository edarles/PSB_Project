#include <PciSphSystem.cuh>
#include <stdio.h>

extern "C"

{
/****************************************************************************************************************************/
/****************************************************************************************************************************/
void pci_SPH_update(double* oldPos, double* pos, double* vel, double* velInterAv, double* velInterAp, 
		    double* restore_oldPos, double* restore_pos, double* restore_velInterAv, double* restore_velInterAp,
		    double* mass, double* radius, double* density, double* restDensity, double* densityError, double* pressure,
		    double* k, double* viscosity, double* l, double* surfaceTension, double threshold, partVoisine voisines,
		    double* normales, double* forcesViscosity, double* forcesSurface, double* forcesPressure, double* forcesAccum,
		    float dt, uint nbBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
  
    compute_Density_PCI<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*) pos, mass, radius, density, restDensity, densityError, k, pressure, voisines);
 
    compute_Viscosity_SurfaceTension_Force_PCI<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*) pos, (double3*) vel, mass, density,  
			       	 radius, viscosity, l, surfaceTension, 
			         (double3*) normales, (double3*) forcesViscosity, (double3*) forcesSurface, (double3*) forcesAccum, voisines);
    uint nbIter = 0;
    while(nbIter < 3) {
        restore_Pos_Vel_PCI<<<numBlocksX,numThreadsX>>>((double3*) restore_velInterAv, (double3*) restore_velInterAp, 
			    (double3*) restore_oldPos, (double3*) restore_pos,
			    (double3*) velInterAv, (double3*) velInterAp, (double3*) oldPos, (double3*) pos, nbBodies);
        cudaDeviceSynchronize();
        compute_Pressure_PCI<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*) pos, mass, radius, k, threshold, 
		             density, restDensity, densityError, pressure, voisines, dt);

	compute_Pressure_Force_PCI<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*) pos, mass, density, densityError, pressure, radius, 
				    (double3*) forcesPressure, threshold, nbIter, voisines);
        
	
	calculate_Dt_CFL<<<1,1>>>(nbBodies, radius, viscosity,
			         (double3*)forcesViscosity, (double3*) forcesSurface, (double3*) forcesPressure, (double3*) forcesAccum,
				 density, dt);

/*	for(unsigned int index=0; index<nbBodies; index++){
		double h = radius[index];
		double3 F = make_double3(forcesAccum[index*3]+ forcesViscosity[index*3] + forcesSurface[index*3] + forcesPressure[index*3],
			    forcesAccum[index*3+1]+ forcesViscosity[index*3+1] + forcesSurface[index*3+1] + forcesPressure[index*3+1],
			    forcesAccum[index*3+2]+ forcesViscosity[index*3+2] + forcesSurface[index*3+2] + forcesPressure[index*3+2]);
		double3 a1 = make_double3(F.x/density[index],F.y/density[index],F.z/density[index]);
		float dt1 = min(pow(h/sqrt(pow(a1.x,2)+pow(a1.y,2)+pow(a1.z,2)),0.5),0.5*h*h/viscosity[index]);
		dt = min(dt,dt1);
	}
	printf("dt:%f\n",dt);
*/
	integrate_SPH_LeapFrog_PCI<<<numBlocksX,numThreadsX>>>((double3*) velInterAv, (double3*) velInterAp, (double3*) oldPos, (double3*) pos, 
				    (double3*) forcesViscosity, (double3*) forcesSurface, (double3*) forcesPressure, (double3*) forcesAccum,
				    density, dt, nbBodies);
        compute_Density_Pressure_PCI<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*) pos, mass, radius, k, 
				     threshold, density, restDensity, densityError, pressure, voisines);

    }
}

/****************************************************************************************************************************/
/****************************************************************************************************************************/
void  evaluateChanges_T_Visc_Rho0_mass(double* pos, double* m_dMass, double* m_interactionRadius, double* kernelParticles,
				       double* m_density, double* m_temperatures, double* m_viscosity, double* m_restDensity,partVoisine voisines,
				       double* m_Dtemperatures, double* m_Dviscosity, double* m_DrestDensity, double *m_Dmass,uint nbParticles)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbParticles,numBlocksX, numThreadsX);
   
    evaluateChanges_T_Visc_Rho0_mass_Kernel<<<numBlocksX,numThreadsX>>>
    ((double3*) pos, m_dMass, m_interactionRadius, m_density, m_temperatures, m_viscosity, voisines,
    m_Dtemperatures, m_Dviscosity, m_Dmass,nbParticles);

    integrateChanges_T_Visc_Rho0_mass_Kernel<<<numBlocksX,numThreadsX>>>
    ((double3*) pos, m_interactionRadius, kernelParticles, m_temperatures, m_viscosity, m_restDensity, m_dMass,
     m_Dtemperatures, m_Dviscosity, m_DrestDensity, m_Dmass, voisines, nbParticles);
}
/****************************************************************************************************************************/
/****************************************************************************************************************************/
}
