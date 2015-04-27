
#include <SphSystem.h>

/*****************************************************************************/
/*****************************************************************************/
SPHSystem::SPHSystem():System()
{
    m_solverIterations = 1;
    dt = 0.01;
    MinS = Vector3(-1,-1,-1);
    MaxS = Vector3(1,1,1);

    m_dPos[0] = m_dPos[1] = 0;
    m_dVel[0] = m_dVel[1] = 0;
    m_hPos[0] = m_hPos[1] = 0;
    m_hVel[0] = m_hVel[1] = 0;
    m_hForce = 0;
    m_dMass = 0;
    m_dParticleRadius = 0;

    m_dVelInterAv = 0;
    m_dVelInterAp = 0;
    m_fPressure = 0;
    m_fViscosity = 0;
    m_fSurface = 0;
    m_normales = 0;
  
    m_interactionRadius = 0;
    m_density = 0;
    m_restDensity = 0;
    m_pressure = 0;
    m_gasStiffness = 0;
    m_threshold = 0;
    m_surfaceTension = 0;
    m_viscosity = 0;
    m_hVelInterAv = 0;
    m_hVelInterAp = 0;

    frameNumber = 0;
    currentTime = 0;
    init();
}

/*****************************************************************************/
SPHSystem::~SPHSystem()
{
   _finalize();
  // #pragma omp parallel for
   for(unsigned int i=0;i<particles.size();i++)
	delete(particles[i]);
   particles.clear();
}

/*****************************************************************************/
/*****************************************************************************/
Vector3 SPHSystem::getMinS()
{
   return MinS;
}
/*****************************************************************************/
Vector3 SPHSystem::getMaxS()
{
   return MaxS;
}
/*****************************************************************************/
void SPHSystem::setMinS(Vector3 MinS)
{
  this->MinS = MinS;
}
/*****************************************************************************/
void SPHSystem::setMaxS(Vector3 MaxS)
{
   this->MaxS = MaxS;
}
/*****************************************************************************/
/*****************************************************************************/
void SPHSystem::init(vector<Particle*> result)
{
     this->particles.clear();
     this->particles.insert(particles.end(),result.begin(),result.end());
     _initialize(0,particles.size());
}
/*****************************************************************************/
void SPHSystem::init()
{
    unsigned int memSize = sizeof(double)*3*maxParticles;
    
    allocateArray((void**)&m_dVel[0], memSize);
    allocateArray((void**)&m_dVel[1], memSize);
    allocateArray((void**)&m_dPos[0], memSize);
    allocateArray((void**)&m_dPos[1], memSize);
    allocateArray((void**)&m_dVelInterAv, memSize);
    allocateArray((void**)&m_dVelInterAp, memSize);

    allocateArray((void**)&m_fPressure, memSize);
    allocateArray((void**)&m_fViscosity, memSize);
    allocateArray((void**)&m_fSurface, memSize);
    allocateArray((void**)&m_normales, memSize);

    allocateArray((void**)&m_dMass, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dParticleRadius, sizeof(double)*maxParticles);

    allocateArray((void**)&m_interactionRadius, sizeof(double)*maxParticles);
    allocateArray((void**)&m_density, sizeof(double)*maxParticles);
    allocateArray((void**)&m_restDensity, sizeof(double)*maxParticles);
    allocateArray((void**)&m_pressure, sizeof(double)*maxParticles);
    allocateArray((void**)&m_gasStiffness, sizeof(double)*maxParticles);
    allocateArray((void**)&m_threshold, sizeof(double)*maxParticles);
    allocateArray((void**)&m_surfaceTension, sizeof(double)*maxParticles);
    allocateArray((void**)&m_viscosity, sizeof(double)*maxParticles);
    allocateArray((void**)&m_wj, sizeof(double)*maxParticles);

    allocateArray((void**)&voisines.nbVoisines,maxParticles*sizeof(int));
    allocateArray((void**)&voisines.listeVoisine,maxParticles*200*sizeof(int));

    m_hPos[0] = new double[maxParticles*3];
    m_hVel[0] = new double[maxParticles*3];
    m_hPos[1] = new double[maxParticles*3];
    m_hVel[1] = new double[maxParticles*3];
    m_hVelInterAv = new double[maxParticles*3];
    m_hVelInterAp = new double[maxParticles*3];
    m_hForce = new double[maxParticles*3];
    m_hColors = new double[maxParticles*4];   
    m_hMass = new double[maxParticles];
    m_hParticleRadius = new double[maxParticles];
    m_hInteractionRadius = new double[maxParticles];
    m_hDensity = new double[maxParticles];
    m_hRestDensity = new double[maxParticles];
    m_hPressure = new double[maxParticles];
    m_hGasStiffness = new double[maxParticles];
    m_hThreshold = new double[maxParticles];
    m_hSurfaceTension = new double[maxParticles];
    m_hViscosity = new double[maxParticles];
    m_hNormales = new double[maxParticles*3];

    FExt->_initialize(maxParticles);

    gridCreated = false;
}
/*****************************************************************************/
void SPHSystem::_initialize(int begin, int numParticles)
{
   if(numParticles>0){
/*
    unsigned int memSize = sizeof(double)*3*numParticles;

    allocateArray((void**)&m_dVel[0], memSize);
    allocateArray((void**)&m_dVel[1], memSize);
    allocateArray((void**)&m_dPos[0], memSize);
    allocateArray((void**)&m_dPos[1], memSize);
    allocateArray((void**)&m_dVelInterAv, memSize);
    allocateArray((void**)&m_dVelInterAp, memSize);

    allocateArray((void**)&m_fPressure, memSize);
    allocateArray((void**)&m_fViscosity, memSize);
    allocateArray((void**)&m_fSurface, memSize);
    allocateArray((void**)&m_normales, memSize);

    allocateArray((void**)&m_dMass, sizeof(double)*numParticles);
    allocateArray((void**)&m_dParticleRadius, sizeof(double)*numParticles);

    allocateArray((void**)&m_interactionRadius, sizeof(double)*numParticles);
    allocateArray((void**)&m_density, sizeof(double)*numParticles);
    allocateArray((void**)&m_restDensity, sizeof(double)*numParticles);
    allocateArray((void**)&m_pressure, sizeof(double)*numParticles);
    allocateArray((void**)&m_gasStiffness, sizeof(double)*numParticles);
    allocateArray((void**)&m_threshold, sizeof(double)*numParticles);
    allocateArray((void**)&m_surfaceTension, sizeof(double)*numParticles);
    allocateArray((void**)&m_viscosity, sizeof(double)*numParticles);
    allocateArray((void**)&m_wj, sizeof(double)*numParticles);

    allocateArray((void**)&voisines.nbVoisines,numParticles*sizeof(int));
    allocateArray((void**)&voisines.listeVoisine,numParticles*100*sizeof(int));

    m_hPos[0] = new double[numParticles*3];
    m_hVel[0] = new double[numParticles*3];
    m_hPos[1] = new double[numParticles*3];
    m_hVel[1] = new double[numParticles*3];
    m_hVelInterAv = new double[numParticles*3];
    m_hVelInterAp = new double[numParticles*3];
    m_hForce = new double[numParticles*3];
    m_hColors = new double[numParticles*4];   
    m_hMass = new double[numParticles];
    m_hParticleRadius = new double[numParticles];
    m_hInteractionRadius = new double[numParticles];
    m_hDensity = new double[numParticles];
    m_hRestDensity = new double[numParticles];
    m_hPressure = new double[numParticles];
    m_hGasStiffness = new double[numParticles];
    m_hThreshold = new double[maxParticles];
    m_hSurfaceTension = new double[numParticles];
    m_hViscosity = new double[numParticles];
    m_hNormales = new double[numParticles*3];

    FExt->_initialize(numParticles);

    gridCreated = false;
*/
    if(begin>0){
    	copyArrayFromDevice(m_hPos[0], m_dPos[0], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVel[0], m_dVel[0], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hPos[1], m_dPos[1], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVel[1], m_dVel[1], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVelInterAv, m_dVelInterAv, 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVelInterAp, m_dVelInterAp, 0, sizeof(double)*3*begin);
    }
    printf("begin:%d numParticles:%d\n",begin, numParticles);
   
    #pragma omp parallel for
    for(int i=begin;i<numParticles;i++){
	 SPHParticle* p   = (SPHParticle*) particles[i];
	
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
	 m_hColors[(i*4)+3] = 1.0;

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

    if(gridCreated==false){
	SPHParticle* p   = (SPHParticle*) particles[0];
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
void SPHSystem::_finalize()
{
  if(particles.size()>0)
  {
    delete[] m_hVel[0];
    delete[] m_hPos[0];
    delete[] m_hVel[1];
    delete[] m_hPos[1];
    delete[] m_hForce;
    delete[] m_hColors;
    delete[] m_hVelInterAv;
    delete[] m_hVelInterAp;
    delete[] m_hMass;
    delete[] m_hParticleRadius;
    delete[] m_hInteractionRadius;
    delete[] m_hDensity;
    delete[] m_hRestDensity;
    delete[] m_hPressure;
    delete[] m_hGasStiffness;
    delete[] m_hThreshold;
    delete[] m_hSurfaceTension;
    delete[] m_hViscosity;
    delete[] m_hNormales;

    freeArray(m_dPos[0]);
    freeArray(m_dPos[1]);
    freeArray(m_dVel[0]);
    freeArray(m_dVel[1]);
    freeArray(m_dMass);
    freeArray(m_dParticleRadius);

    freeArray(m_dVelInterAv);
    freeArray(m_dVelInterAp);

    freeArray(m_fPressure);
    freeArray(m_fViscosity);
    freeArray(m_fSurface);
    freeArray(m_normales);

    freeArray(m_interactionRadius);
    freeArray(m_density);
    freeArray(m_restDensity);
    freeArray(m_pressure);
    freeArray(m_gasStiffness);
    freeArray(m_threshold);
    freeArray(m_surfaceTension);
    freeArray(m_viscosity);
    freeArray(m_wj);

    freeArray(voisines.listeVoisine);
    freeArray(voisines.nbVoisines);   

    FExt->_finalize();

    delete(grid);
  }
}
/*****************************************************************************/
/*****************************************************************************/
void SPHSystem::emitParticles()
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
void SPHSystem::update()
{
    // emit particles
    emitParticles();

    reinitCells(grid);
    storeParticles(grid,m_dPos[1],particles.size());
    searchNeighbooring(m_dPos[1],m_interactionRadius, grid, 1, voisines, particles.size());

    // evaluate densities, normales and pressure, viscosity and surface tension forces
    evaluateDensitiesForces();
    
    // evaluate forces
    evaluateForcesExt();
	
    // calculate time step (CFL condition)
    double timeStep = calculateTimeStep();
    dt = timeStep;

    // integrate
    integrate();

    // Interpolate velocities
    interpolateVelocities();

    // Collision
    collide();

    t++;
}
/*****************************************************************************/
/*****************************************************************************/
void SPHSystem::evaluateDensitiesForces()
{
    evaluate_densities_forces(m_dPos[1],m_dVel[1],m_dMass, m_interactionRadius, m_density, m_pressure,
			      m_normales,m_restDensity,m_viscosity,m_gasStiffness,m_threshold,m_surfaceTension,
 			      particles.size(),m_fPressure, m_fViscosity, m_fSurface, FExt->m_F, voisines);
}
/*****************************************************************************/
/*****************************************************************************/
void SPHSystem::evaluateForcesExt()
{  
   FExt->evaluate(m_dPos[1],FExt->m_F, m_density, particles.size());
}
/*****************************************************************************/
/*****************************************************************************/
void SPHSystem::collide()
{
    if(collision->getObjects().size()>0){
	collision->collide(m_dPos[0],m_dPos[1],m_dVel[0], m_dVel[1], 
			   particles[0]->getParticleRadius(), dt, particles.size());
	postProcessCollide(m_dVelInterAv, m_dVelInterAp, m_dVel[0], m_dVel[1], particles.size());
   }
}
/*****************************************************************************/
/*****************************************************************************/
void SPHSystem::integrate()
{
    integrateSPHSystem(m_dVelInterAv, m_dVelInterAp, m_dPos[0], m_dPos[1], FExt->m_F, m_density, dt, particles.size());
    copyArrayFromDevice(m_hPos[1], m_dPos[1], 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(m_hPos[0], m_dPos[0], 0, sizeof(double)*3*particles.size());
}
/*****************************************************************************/
/*****************************************************************************/
void SPHSystem::interpolateVelocities()
{
    interpolateSPHVelocities(m_dVelInterAv, m_dVelInterAp, m_dVel[0], m_dVel[1], particles.size());

    copyArrayFromDevice(m_hPos[1], m_dPos[1], 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(m_hPos[0], m_dPos[0], 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(m_hVel[0], m_dVel[0], 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(m_hVel[1], m_dVel[1], 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(m_hVelInterAp, m_dVelInterAp, 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(m_hVelInterAv, m_dVelInterAv, 0, sizeof(double)*3*particles.size());
}
/*****************************************************************************/ 
/*****************************************************************************/
double SPHSystem::calculateTimeStep()
{
    double* acc = new double[3*particles.size()];
    double* dens = new double[particles.size()];
    copyArrayFromDevice(m_hVel[1], m_dVel[1], 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(acc, FExt->m_F, 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(dens, m_density, 0, sizeof(double)*particles.size());
    double vmax = 0; double amax = 0; double hmin = 1000;
    for(int i=0;i<particles.size();i++){
	Vector3 v = Vector3(m_hVel[1][(i*3)],m_hVel[1][(i*3)+1],m_hVel[1][(i*3)+2]);
	Vector3 a = Vector3(acc[i*3]/dens[i],acc[(i*3)+1]/dens[i],acc[(i*3)+2]/dens[i]);
	double lv = v.length();
	double la = a.length();
	if(lv>vmax) vmax = lv;
	if(la>amax) amax = la;
	if(m_hInteractionRadius[i]<hmin) hmin = m_hInteractionRadius[i];
    }
    delete[] acc;
    delete[] dens;
    double a = vmax*vmax+4*amax*0.5*hmin;
    return -(vmax-sqrt(a))/(2*amax);
}
/*****************************************************************************/ 
/*****************************************************************************/
void SPHSystem::evaluateIso(double* pos, uint nbCellsX, uint nbCellsY, uint nbCellsZ, double scale)
{
	reinitCells(grid);
        storeParticles(grid,m_dPos[1],particles.size());
	evaluateIso_CUDA(pos,nbCellsX,nbCellsY,nbCellsZ,scale,grid,m_dPos[1],m_interactionRadius,particles.size());
}
/*****************************************************************************/
void SPHSystem::computeNormales_Vertex(double* posV, double* normales, float scale, uint nbV)
{
	computeNormales_Vertex_CUDA (posV, normales, scale, nbV,
			            grid, m_dPos[1], m_interactionRadius, m_dMass, m_density, particles.size());
}
/*****************************************************************************/
/*****************************************************************************/
void SPHSystem::displayParticles(ParticleDisplay mode, Vector3 color)
{
	if(particles.size()>0){
		System::displayParticles(mode,color);
		//if(gridCreated)
		//	displayGridByIso(&grid);
		//	displayGrid(&grid);
	}
}
/*****************************************************************************/
void SPHSystem::displayParticlesByField(uint field)
{
    // DISPLAY BY DENSITY FIELD
    if(field==0){
    	copyArrayFromDevice(m_hDensity,m_density,0,sizeof(double)*particles.size());
	double dMax = 0; double dMin = 10000;
	for(int i=0;i<particles.size();i++){
		if(m_hDensity[i]>dMax) dMax = m_hDensity[i];
		if(m_hDensity[i]<dMin) dMin = m_hDensity[i];
	}
	//printf("dMin:%f dMax:%f\n",dMin, dMax);
	#pragma omp parallel for
    	for(uint i=0;i<particles.size();i++){
		float hue;
		if(dMax==dMin) hue = 240;
		else hue = 240 * (dMax - min((double)m_hDensity[i],dMax)) / (dMax-dMin);
		Vector3 Hsv(hue,1,1);
		Vector3 Rgb = convertHsvToRgb(Hsv);
		m_hColors[i*4] = Rgb.x();
		m_hColors[i*4+1] = Rgb.y();
		m_hColors[i*4+2] = Rgb.z();
		m_hColors[i*4+3] = 0.5;
    	}
    }
    // DISPLAY BY PRESSURE FIELD
    if(field==1){
    	copyArrayFromDevice(m_hPressure,m_pressure,0,sizeof(double)*particles.size());
	double pMax = -1000000; double pMin = 1000000;
	for(int i=0;i<particles.size();i++){
		if(m_hPressure[i]>pMax) pMax = m_hPressure[i];
		if(m_hPressure[i]<pMin) pMin = m_hPressure[i];
	}
	//double pMin = 0; double pMax = 100000;
	//printf("pMin:%f pMax:%f\n",pMin, pMax);
	#pragma omp parallel for
    	for(uint i=0;i<particles.size();i++){
		float hue = 240 * (pMax - min(fabs((double)m_hPressure[i]),pMax)) / (pMax-pMin);
		if(pMax==pMin) hue = 240;
		Vector3 Hsv(hue,1,1);
		Vector3 Rgb = convertHsvToRgb(Hsv);
		m_hColors[i*4] = Rgb.x();
		m_hColors[i*4+1] = Rgb.y();
		m_hColors[i*4+2] = Rgb.z();
    	}
   }
   // DISPLAY BY VISCOSITY FIELD
   if(field==2){
    	copyArrayFromDevice(m_hViscosity,m_viscosity,0,sizeof(double)*particles.size());
	double vMin = 3.5; double vMax = 10;
	#pragma omp parallel for
    	for(uint i=0;i<particles.size();i++){
		float hue = 240 * (vMax - min((double)m_hViscosity[i],vMax)) / (vMax-vMin);
		Vector3 Hsv(hue,1,1);
		Vector3 Rgb = convertHsvToRgb(Hsv);
		m_hColors[i*4] = Rgb.x();
		m_hColors[i*4+1] = Rgb.y();
		m_hColors[i*4+2] = Rgb.z();
    	}
   }
   //DISPLAY BY SURFACE TENSION FIELD
   if(field==3){
    	copyArrayFromDevice(m_hSurfaceTension,m_fSurface,0,sizeof(double)*3*particles.size());
	#pragma omp parallel for
    	for(uint i=0;i<particles.size();i++){
		double sTension = sqrt(m_hSurfaceTension[i*3]*m_hSurfaceTension[i*3] + 
				       m_hSurfaceTension[i*3+1]*m_hSurfaceTension[i*3+1] + 
				       m_hSurfaceTension[i*3+2]*m_hSurfaceTension[i*3+2]);
		if(sTension!=0) {
			m_hColors[i*4] = 255;
			m_hColors[i*4+1] = 0;
			m_hColors[i*4+2] = 0;
		}
		else {
			m_hColors[i*4] = 0;
			m_hColors[i*4+1] = 0;
			m_hColors[i*4+2] = 255;
		}
    	}
   }
   //DISPLAY BY VELOCITY FIELD
   if(field==3){
    	copyArrayFromDevice(m_hVel[1],m_dVel[1],0,sizeof(double)*3*particles.size());
	double vmax, vmin;
	vmax = 0; vmin = 1000000;
	for(uint i=0;i<particles.size();i++){
		double length = sqrt(m_hVel[1][i*3]*m_hVel[1][i*3]+m_hVel[1][i*3+1]*m_hVel[1][i*3+1]+m_hVel[1][i*3+2]*m_hVel[1][i*3+2]);
		if(vmax<length) vmax = length;
		if(vmin>length) vmin = length;
	}
	#pragma omp parallel for
    	for(uint i=0;i<particles.size();i++){
		double length = sqrt(m_hVel[1][i*3]*m_hVel[1][i*3]+m_hVel[1][i*3+1]*m_hVel[1][i*3+1]+m_hVel[1][i*3+2]*m_hVel[1][i*3+2]);
		float hue = 240 * (vmax - min(length,vmax)) / (vmax-vmin);
		Vector3 Hsv(hue,1,1);
		Vector3 Rgb = convertHsvToRgb(Hsv);
		m_hColors[i*4] = Rgb.x();
		m_hColors[i*4+1] = Rgb.y();
		m_hColors[i*4+2] = Rgb.z();
    	}
   }
}
/*****************************************************************************/
/*****************************************************************************/
Vector3 SPHSystem::convertHsvToRgb(Vector3 Hsv)
{
   Vector3 result;
   int t = (int)(floor(Hsv.x()/60))%6;
   double f = (Hsv.x()/60)-t;
   double l = Hsv.z()*(1-Hsv.y());
   double m = Hsv.z()*(1-f*Hsv.y());
   double n = Hsv.z()*(1-(1-f)*Hsv.y());
   if(t==0) {result.setX(Hsv.z()); result.setY(n); result.setZ(l);};
   if(t==1) {result.setX(m); result.setY(Hsv.z()); result.setZ(l);};
   if(t==2) {result.setX(l); result.setY(Hsv.z()); result.setZ(n);};
   if(t==3) {result.setX(l); result.setY(m); result.setZ(Hsv.z());};
   if(t==4) {result.setX(n); result.setY(l); result.setZ(Hsv.z());};
   if(t==5) {result.setX(Hsv.z()); result.setY(l); result.setZ(m);};
   return result;
}
