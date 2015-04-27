#include <PciSphSystem.cuh>
#include <stdio.h>

extern "C"
{
/**********************************************************************************************************/
void evaluate_densities_forcesVisc_PCI(double* pos, double* vel, double* mass, double* radius, double* densities, 
			       double* pressure, double* normales, double* restDensities, double* densityError, double* viscosities, 
			       double* threshold, double* surfaceTension, 
		               int numBodies, double* fViscosity, double* fSurface, double* fPressure, partVoisine voisines)
{
    int numThreadsX, numBlocksX;
    computeGridSize(numBodies,numBlocksX, numThreadsX);

    compute_Density_PCI<<<numBlocksX,numThreadsX>>>(numBodies, (double3*) pos, mass, radius, densities, restDensities, densityError, pressure, voisines);

    compute_Viscosity_SurfaceTension_Force_PCI<<<numBlocksX,numThreadsX>>>(numBodies, (double3*) pos, (double3*) vel, mass, densities,  
			       						   pressure, radius, viscosities, threshold, surfaceTension, 
			         					   (double3*) normales, (double3*) fViscosity, (double3*) fSurface, (double3*) fPressure, voisines);

    cudaDeviceSynchronize();
}

/**********************************************************************************************************/
void integrate_PCI_SPHSystem(double* velAv, double *velAp, double* posAv, double* posAp, double* fV, double* fS, double* fP, double* fExt, double* densities, double dt, int nbBodies)
{
    int numThreadsX, numBlocksX;
    computeGridSize(nbBodies,numBlocksX, numThreadsX);
    integrateSPH_LeapFrog_Forces<<<numBlocksX,numThreadsX>>>((double3*) velAv, (double3*) velAp, (double3*) posAv, (double3*) posAp, 
				      (double3*) fV, (double3*) fS, (double3*) fP, (double3*) fExt,
				      densities, dt, nbBodies);

    cudaDeviceSynchronize();
}
/****************************************************************************************************************************/
void pci_SPH_pressureForce(double* oldPos, double* pos, double* vel, double* velInterAv, double* velInterAp, 
		    double* mass, double* radius, double* density, double* restDensity, double* densityError, double* k, double* pressure,
		    double threshold, partVoisine voisines,double* forcesPressure, float dt, uint nbBodies)
{
    	int numThreadsX, numBlocksX;
    	computeGridSize(nbBodies,numBlocksX, numThreadsX);
	
	// On calcul la nouvelle densité et la différence entre cette densité et la densité au repos
	compute_Density_PCI_predict<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*) pos, mass, radius, density, restDensity, densityError, voisines);
	cudaDeviceSynchronize();

	// On calcul la pression (si l'erreur commise sur la densité est inférieur à un certain pourcentage)
        compute_Pressure_PCI<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*) pos, mass, radius, threshold, 
		             				 density, restDensity, densityError, k, pressure, voisines, dt);
	cudaDeviceSynchronize();

	// On calcul la force de pression (si l'erreur commise sur la densité est inférieur à un certain pourcentage)
	compute_Pressure_Force_PCI<<<numBlocksX,numThreadsX>>>(nbBodies, (double3*) pos, mass, 
				     density, restDensity, densityError, pressure, radius, 
				    (double3*) forcesPressure, threshold, voisines);
        cudaDeviceSynchronize();
}

}
