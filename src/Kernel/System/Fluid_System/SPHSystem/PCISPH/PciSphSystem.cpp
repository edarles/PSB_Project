
#include <PciSphSystem.h>
#include <PciSphSystem.cuh>

/*****************************************************************************/
/*****************************************************************************/
PCI_SPHSystem::PCI_SPHSystem():SPHSystem()
{
   	unsigned int memSize = sizeof(double)*3*maxParticles;
    
        allocateArray((void**)&m_dPosPredict[0], memSize);
        allocateArray((void**)&m_dPosPredict[1], memSize);
	allocateArray((void**)&m_dVelPredict[0], memSize);
        allocateArray((void**)&m_dVelPredict[1], memSize);
	allocateArray((void**)&m_dVelInterPredict[0], memSize);
        allocateArray((void**)&m_dVelInterPredict[1], memSize);
	allocateArray((void**)&m_densityError, sizeof(double)*maxParticles);
	allocateArray((void**)&m_densityPredict, sizeof(double)*maxParticles);

	m_hDensityCorrected = new double[maxParticles];
	m_hErrorDensity = new double[maxParticles];
        errorThreshold = 10;
}

/*****************************************************************************/
PCI_SPHSystem::~PCI_SPHSystem()
{
	freeArray(m_dPosPredict[0]);
        freeArray(m_dPosPredict[1]);
	freeArray(m_dVelPredict[0]);
        freeArray(m_dVelPredict[1]);
	freeArray(m_dVelInterPredict[0]);
        freeArray(m_dVelInterPredict[1]);
	freeArray(m_densityError);
	freeArray(m_densityPredict);

	delete[] m_hDensityCorrected;
	delete[] m_hErrorDensity;
}

/*****************************************************************************/
void PCI_SPHSystem::init(vector<Particle*> particles)
{
      this->particles.clear();
      for(unsigned int i=0;i<particles.size();i++)
		this->particles.push_back(particles[i]);
      _initialize(0,particles.size());      
}

/*****************************************************************************/
void PCI_SPHSystem::_initialize(int begin, int numParticles)
{
   if(numParticles>0){
    unsigned int N = numParticles - begin;
    if(begin>0){
    	copyArrayFromDevice(m_hPos[0], m_dPos[0], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVel[0], m_dVel[0], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hPos[1], m_dPos[1], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVel[1], m_dVel[1], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVelInterAv, m_dVelInterAv, 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVelInterAp, m_dVelInterAp, 0, sizeof(double)*3*begin);
    }
    for(unsigned int i=begin;i<numParticles;i++){
	 PCI_SPHParticle* p   = (PCI_SPHParticle*) particles[i];
	
	 m_hPos[0][i*3]   = p->getOldPos().x();
	 m_hPos[0][(i*3)+1] = p->getOldPos().y();
	 m_hPos[0][(i*3)+2] = p->getOldPos().z();

         m_hVel[0][i*3]   = p->getOldVel().x();
	 m_hVel[0][(i*3)+1] = p->getOldVel().y();
	 m_hVel[0][(i*3)+2] = p->getOldVel().z();

         m_hPos[1][(i*3)]   = p->getNewPos().x();
	 m_hPos[1][(i*3)+1] = p->getNewPos().y();
	 m_hPos[1][(i*3)+2] = p->getNewPos().z();

         m_hVel[1][i*3]   = p->getNewVel().x();
	 m_hVel[1][(i*3)+1] = p->getNewVel().y();
	 m_hVel[1][(i*3)+2] = p->getNewVel().z();

	 m_hVelInterAv[i*3] = p->getVelInterAv().x();
	 m_hVelInterAv[(i*3)+1] = p->getVelInterAv().y();
	 m_hVelInterAv[(i*3)+2] = p->getVelInterAv().z();

	 m_hVelInterAp[i*3] = p->getVelInterAp().x();
	 m_hVelInterAp[(i*3)+1] = p->getVelInterAp().y();
	 m_hVelInterAv[(i*3)+2] = p->getVelInterAp().z();

	 m_hMass[i] = p->getMass();
	 m_hParticleRadius[i] = p->getParticleRadius();

         m_hInteractionRadius[i] = p->getInteractionRadius();
         m_hDensity[i] = 0;
	 m_hRestDensity[i] = p->getRestDensity();
         m_hPressure[i] = 0;
         m_hGasStiffness[i] = p->getGasStiffness();
         m_hThreshold[i] = p->getThreshold();
         m_hSurfaceTension[i] = p->getSurfaceTension();
         m_hViscosity[i] = p->getViscosity();
	 m_hNormales[i*3]=0; m_hNormales[(i*3)+1]=0; m_hNormales[(i*3)+2]=0;

	 m_hColors[i*4] = p->getColor().x();
	 m_hColors[(i*4)+1] = p->getColor().y();
	 m_hColors[(i*4)+2] = p->getColor().z();
	 m_hColors[(i*4)+3] = 1;
   }
    copyArrayToDevice(m_dPos[0], m_hPos[0], 0, sizeof(double)*3*numParticles);
    copyArrayToDevice(m_dVel[0], m_hVel[0], 0, sizeof(double)*3*numParticles);
    copyArrayToDevice(m_dPos[1], m_hPos[1], 0, sizeof(double)*3*numParticles);
    copyArrayToDevice(m_dVel[1], m_hVel[1], 0, sizeof(double)*3*numParticles);
    copyArrayToDevice(m_dVelInterAv, m_hVelInterAv, 0, sizeof(double)*3*numParticles);
    copyArrayToDevice(m_dVelInterAp, m_hVelInterAp, 0, sizeof(double)*3*numParticles);

    copyArrayToDevice(m_dMass, m_hMass, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_dParticleRadius, m_hParticleRadius, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_interactionRadius, m_hInteractionRadius, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_restDensity, m_hRestDensity, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_gasStiffness, m_hGasStiffness, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_threshold, m_hThreshold, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_surfaceTension, m_hSurfaceTension, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_viscosity, m_hViscosity, 0, sizeof(double)*numParticles);
  
    double* F = new double[3*numParticles];
    #pragma omp parallel for
    for(uint i=0;i<numParticles;i++){
		F[i*3] = 0;
		F[i*3+1] = 0;
		F[i*3+2] = 0;
     }
     copyArrayToDevice(m_fViscosity,F,0,sizeof(double)*3*numParticles);
     copyArrayToDevice(m_fSurface,F,0,sizeof(double)*3*numParticles);
     copyArrayToDevice(m_fPressure,F,0,sizeof(double)*3*numParticles);
     delete[] F;

    if(gridCreated==false){
	PCI_SPHParticle* p   = (PCI_SPHParticle*) particles[0];
	Vector3 C = Vector3((MaxS.x()+MinS.x())/2,(MaxS.y()+MinS.y())/2,(MaxS.z()+MinS.z())/2);
	double l = MaxS.x()-MinS.x(); 
	double w = MaxS.y()-MinS.y(); 
	double d = MaxS.z()-MinS.z(); 
	grid = (UniformGrid*)malloc(sizeof(UniformGrid));
	createGrid(C.x(),C.y(),C.z(),l,w,d,p->getInteractionRadius(),grid);
        gridCreated = true;
	displayGrid(grid);
    }
 
  }
}

/*****************************************************************************/
/*****************************************************************************/
void PCI_SPHSystem::emitParticles()
{
    vector<Particle*> result = emitters->emitParticles();
    if(result.size()>0){
	    unsigned int nbAv = particles.size();
	    particles.insert(particles.end(),result.begin(),result.end());
    	    _initialize(nbAv,particles.size());
    }
}
/*****************************************************************************/
/*****************************************************************************/
void PCI_SPHSystem::update()
{
    // emit particles
    emitParticles();
   
    reinitCells(grid);
    storeParticles(grid,m_dPos[1],particles.size());
    searchNeighbooring(m_dPos[1],m_interactionRadius, grid, 1, voisines, particles.size());

    evaluate_densities_forcesVisc_PCI(m_dPos[1], m_dVel[1], m_dMass, m_interactionRadius, m_density, 
	m_pressure, m_normales, m_restDensity, m_densityError, m_viscosity, 
	m_threshold, m_surfaceTension, particles.size(), m_fViscosity, m_fSurface, m_fPressure, voisines);


    FExt->init(particles.size());
    evaluateForcesExt();

    copyArrayDeviceToDevice(m_dPosPredict[0], m_dPos[0], 0, sizeof(double)*3*particles.size());
    copyArrayDeviceToDevice(m_dPosPredict[1], m_dPos[1], 0, sizeof(double)*3*particles.size());
    copyArrayDeviceToDevice(m_dVelPredict[0], m_dVel[0], 0, sizeof(double)*3*particles.size());
    copyArrayDeviceToDevice(m_dVelPredict[1], m_dVel[1], 0, sizeof(double)*3*particles.size());
    copyArrayDeviceToDevice(m_dVelInterPredict[0], m_dVelInterAv, 0, sizeof(double)*3*particles.size());
    copyArrayDeviceToDevice(m_dVelInterPredict[1], m_dVelInterAp, 0, sizeof(double)*3*particles.size());

    copyArrayToDevice(m_densityPredict,m_density,0,sizeof(double)*particles.size());

    int iter = 0;
    double errorTotal = errorThreshold+1;
   
    while(iter<3 && errorTotal>errorThreshold){

		// Integration
		 integrate_PCI_SPHSystem(m_dVelInterPredict[0], m_dVelInterPredict[1], m_dPosPredict[0], m_dPosPredict[1], m_fViscosity, m_fSurface, m_fPressure,
		 FExt->m_F, m_densityPredict, dt, particles.size());
 	         interpolateSPHVelocities(m_dVelInterPredict[0], m_dVelInterPredict[1], m_dVelPredict[0], m_dVelPredict[1], particles.size());

		 reinitCells(grid);
    		 storeParticles(grid,m_dPosPredict[1],particles.size());
    		 searchNeighbooring(m_dPosPredict[1],m_interactionRadius, grid, 1, voisines, particles.size());

		// Corrige Pressure and calculate pressure forces
		pci_SPH_pressureForce(m_dPosPredict[0],m_dPosPredict[1],m_dVelPredict[0],m_dVelInterPredict[0], m_dVelInterPredict[1], 
				  m_dMass, m_interactionRadius, m_densityPredict, m_restDensity, m_densityError, m_gasStiffness, m_pressure, 
				  errorThreshold, voisines, m_fPressure, dt, particles.size());

		iter++;
		copyArrayFromDevice(m_hErrorDensity, m_densityError, 0, sizeof(double)*particles.size());
		errorTotal = 0;
		for(int i=0;i<particles.size();i++)
			errorTotal += fabs(m_hErrorDensity[i]);
		errorTotal /= particles.size();
		//printf("errorTotal:%f\n",errorTotal);
   }

   // Calculate time step
   double timeStep = dt;//calculateTimeStep();
   //double timeStep = dt;//min(dt, timeStep);

   // Integration
   integrate_PCI_SPHSystem(m_dVelInterAv, m_dVelInterAp, m_dPos[0], m_dPos[1], m_fViscosity, m_fSurface, m_fPressure, FExt->m_F, m_densityPredict, timeStep, particles.size());
   interpolateSPHVelocities(m_dVelInterAv, m_dVelInterAp, m_dVel[0], m_dVel[1], particles.size());

   copyArrayFromDevice(m_hPos[1], m_dPos[1], 0, sizeof(double)*3*particles.size());
   copyArrayFromDevice(m_hPos[0], m_dPos[0], 0, sizeof(double)*3*particles.size());
   copyArrayFromDevice(m_hVel[0], m_dVel[0], 0, sizeof(double)*3*particles.size());
   copyArrayFromDevice(m_hVel[1], m_dVel[1], 0, sizeof(double)*3*particles.size());
   copyArrayFromDevice(m_hVelInterAp, m_dVelInterAp, 0, sizeof(double)*3*particles.size());
   copyArrayFromDevice(m_hVelInterAv, m_dVelInterAv, 0, sizeof(double)*3*particles.size());

   collide();


   // Store time step per frame
   /*FILE *f = fopen("timeStepPCI.dat","a");
   fprintf(f,"%d %f\n",frameNumber,timeStep);
   fclose(f);
   frameNumber++;
*/
    // Calcul density error avg
    errorDensityAvg = 0;
    copyArrayFromDevice(m_hDensity,m_densityPredict,0,sizeof(double)*particles.size());
    for(unsigned int i=0;i<particles.size();i++)
	errorDensityAvg += fabs(m_hDensity[i]-m_hRestDensity[i])/m_hRestDensity[i];
    errorDensityAvg /= particles.size();
    FILE *f2 = fopen("errorDensityPCI.dat","a");
    fprintf(f2,"%f %f\n",currentTime, errorDensityAvg);
    fclose(f2);
    currentTime += timeStep;

}
/*****************************************************************************/ 
/*****************************************************************************/
double PCI_SPHSystem::calculateTimeStep()
{
    double* fext = new double[3*particles.size()];
    double* fpress = new double[3*particles.size()];
    double* fvisc = new double[3*particles.size()];
    double* fsurf = new double[3*particles.size()];
    double* dens = new double[particles.size()];

    copyArrayFromDevice(m_hVel[1], m_dVel[1], 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(fext, FExt->m_F, 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(fpress, m_fPressure, 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(fvisc, m_fViscosity, 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(fsurf, m_fSurface, 0, sizeof(double)*3*particles.size());
 
    copyArrayFromDevice(dens, m_densityPredict, 0, sizeof(double)*particles.size());
    double vmax = 0; double amax = 0; double hmin = 1000;
    for(int i=0;i<particles.size();i++){
	Vector3 v = Vector3(m_hVel[1][(i*3)],m_hVel[1][(i*3)+1],m_hVel[1][(i*3)+2]);
	Vector3 a = Vector3((fext[i*3]+fpress[i*3]+fvisc[i*3]+fsurf[i*3])/dens[i],
			    (fext[i*3+1]+fpress[i*3+1]+fvisc[i*3+1]+fsurf[i*3+1])/dens[i],
			    (fext[i*3+2]+fpress[i*3+2]+fvisc[i*3+2]+fsurf[i*3+2])/dens[i]);
	double lv = v.length();
	double la = a.length();
	if(lv>vmax) vmax = lv;
	if(la>amax) amax = la;
	if(m_hInteractionRadius[i]<hmin) hmin = m_hInteractionRadius[i];
    }
    delete[] fext;
    delete[] fpress;
    delete[] fvisc;
    delete[] fsurf;
    delete[] dens;

    double a = vmax*vmax+4*amax*0.5*hmin;
    return -(vmax-sqrt(a))/(2*amax);
}
/*****************************************************************************/ 
/*****************************************************************************/
