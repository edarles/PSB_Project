#include <MSphSystem.h>
#include <QColor>
#include <MSphSystem.cuh>
/*****************************************************************************/
/*****************************************************************************/
MSPHSystem::MSPHSystem():SPHSystem()
{
  init();
  phases.clear();
}
/*****************************************************************************/
MSPHSystem::~MSPHSystem()
{
   for(unsigned int i=0;i<phases.size();i++)
	delete (phases[i]);
   phases.clear();
}
/*****************************************************************************/
/*****************************************************************************/
void MSPHSystem::init()
{
    SPHSystem::init();
    
/**************************************************************************************************************************************/
/**************************************************************************************************************************************/
    phases.clear();

    allocateArray((void**)&m_d_T0k, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_dTk, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_D0k, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_dDk, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_Dk, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_Tk, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_Mu0k, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_dMuk, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_dAlphak, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_alphak, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_densRestk, sizeof(double)*maxParticles*maxPhases);
    allocateArray((void**)&m_d_T, sizeof(double)*maxParticles);
    allocateArray((void**)&m_d_D, sizeof(double)*maxParticles);
    allocateArray((void**)&m_d_partPhases, sizeof(int)*maxParticles);

    m_h_T0k = new double[maxParticles*maxPhases];
    m_h_dTk = new double[maxParticles*maxPhases];
    m_h_D0k = new double[maxParticles*maxPhases];
    m_h_dDk = new double[maxParticles*maxPhases];
    m_h_Dk = new double[maxParticles*maxPhases];
    m_h_Tk = new double[maxParticles*maxPhases];
    m_h_Mu0k = new double[maxParticles*maxPhases];
    m_h_dMuk = new double[maxParticles*maxPhases];
    m_h_dAlphak = new double[maxParticles*maxPhases];
    m_h_alphak = new double[maxParticles*maxPhases];
    m_h_densRestk = new double[maxParticles*maxPhases];
    m_h_T = new double[maxParticles];
    m_h_D = new double[maxParticles];
    m_h_partPhases = new int[maxParticles];

    tempMin = 1000;
    tempMax = -1000;
    muMin = 1000;
    muMax = -1000;

/**************************************************************************************************************************************/
/**************************************************************************************************************************************/
}

/**************************************************************************************************************************************/
/**************************************************************************************************************************************/
void MSPHSystem::addEmitter(Emitter *E)
{
	emitters->addEmitter(E);
	if(phases.size()<maxPhases){
		SimulationData* data = E->getData();
		if(typeid(*data)==typeid(SimulationData_MSPHSystem)){
			Phase* ph = ((SimulationData_MSPHSystem*) data)->getPhase();
			phases.push_back(ph);
		}
	}
}
	
/*****************************************************************************/
void MSPHSystem::_initialize(int begin, int numParticles)
{
   if(numParticles>0){

    if(begin>0){
    	copyArrayFromDevice(m_hPos[0], m_dPos[0], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVel[0], m_dVel[0], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hPos[1], m_dPos[1], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVel[1], m_dVel[1], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVelInterAv, m_dVelInterAv, 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVelInterAp, m_dVelInterAp, 0, sizeof(double)*3*begin);
	copyArrayFromDevice(m_h_T, m_d_T, 0, sizeof(double)*begin);
	copyArrayFromDevice(m_h_D, m_d_D, 0, sizeof(double)*begin);
	//copyArrayFromDevice(m_h_alphak, m_d_alphak, 0, sizeof(double)*begin*maxPhases);
	copyArrayFromDevice(m_hViscosity, m_viscosity, 0, sizeof(double)*begin);
	//copyArrayFromDevice(m_h_T, m_d_T, 0, sizeof(double)*begin);
    }
   //printf("numPhases:%d\n",phases.size());
  // printf("begin:%d numParticles:%d\n",begin, numParticles);

    for(int i=begin;i<numParticles;i++){
	 MSPHParticle* p   = (MSPHParticle*) particles[i];
	 Phase *ph = p->getPhase();

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
	 m_hColors[(i*4)+3] = 0.1; 

/**************************************************************************************************************************************/
/**************************************************************************************************************************************/
	m_h_T[i] = ph->getTemperature();
	// Ajoute la phase j à la particule i
        for(unsigned int j=0;j<phases.size();j++){
	 	m_h_D0k[i*maxPhases+j] =  phases[j]->getD0();
	 	m_h_T0k[i*maxPhases+j] =  phases[j]->getTemperature();
	 	m_h_Mu0k[i*maxPhases+j] = m_hViscosity[i];
		if(ph==phases[j]){
			m_h_alphak[i*maxPhases+j] = phases[j]->getAlpha();
			m_h_partPhases[i] = j;
		}
		else {
			m_h_alphak[i*maxPhases+j] = 0.0;
		}
		//printf("conc:%d %f\n",j, m_h_alphak[i*maxPhases+j]);
	 	m_h_densRestk[i*maxPhases+j] = m_hRestDensity[i];
         	m_h_D[i] = phases[j]->getD0();
	 	m_h_dDk[i*maxPhases+j] = 0;
	 	m_h_dTk[i*maxPhases+j] = 0;
	 	m_h_dMuk[i*maxPhases+j] = 0;
	 	m_h_Dk[i*maxPhases+j] = phases[j]->getD0() ;
                m_h_Tk[i*maxPhases+j] = ph->getTemperature();//phases[j]->getTemperature() ;
	}
 	if(tempMin>m_h_T[i]) tempMin = m_h_T[i];
	if(tempMax<m_h_T[i]) tempMax = m_h_T[i];

        if(muMin>m_hViscosity[i]) muMin = m_hViscosity[i];
	if(muMax<m_hViscosity[i]) muMax = m_hViscosity[i];
	//printf("muMin:%f muMax:%f\n",muMin, muMax);
/**************************************************************************************************************************************/
/**************************************************************************************************************************************/
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

    copyArrayToDevice(m_d_T, m_h_T, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_d_D, m_h_D, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_d_D0k, m_h_D0k, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_Dk, m_h_Dk, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_Tk, m_h_Tk, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_T0k, m_h_T0k, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_Mu0k, m_h_Mu0k, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_alphak, m_h_alphak, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_densRestk, m_h_densRestk, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_dDk, m_h_dDk, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_dMuk, m_h_dMuk, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_dTk, m_h_dTk, 0, sizeof(double)*numParticles*maxPhases);
    copyArrayToDevice(m_d_partPhases, m_h_partPhases, 0, sizeof(int)*numParticles);

    if(gridCreated==false){
	MSPHParticle* p   = (MSPHParticle*) particles[0];
	Vector3 C = Vector3((MaxS.x()+MinS.x())/2,(MaxS.y()+MinS.y())/2,(MaxS.z()+MinS.z())/2);
	double l = MaxS.x()-MinS.x(); 
	double w = MaxS.y()-MinS.y(); 
	double d = MaxS.z()-MinS.z(); 
	grid = (UniformGrid*)malloc(sizeof(UniformGrid));
	createGrid(C.x(),C.y(),C.z(),l,w,d,p->getInteractionRadius(),grid);
        gridCreated = true;
	displayGrid(grid);
    }
 

    //for(int i=0;i<particles.size();i++)
	//printf("index:%d phase:%d\n",i,m_h_partPhases[i]);
  }
}
/*****************************************************************************/
void MSPHSystem::_finalize()
{
  if(particles.size()>0)
  {
    SPHSystem::_finalize();

    freeArray(m_d_T0k); 
    freeArray(m_d_D0k); 
    freeArray(m_d_dTk); 
    freeArray(m_d_dDk); 
    freeArray(m_d_Dk);
    freeArray(m_d_Tk); 
    freeArray(m_d_Mu0k); 
    freeArray(m_d_dMuk);
    freeArray(m_d_dAlphak);  
    freeArray(m_d_alphak); 
    freeArray(m_d_densRestk); 
    freeArray(m_d_T); 
    freeArray(m_d_D); 
    freeArray(m_d_partPhases); 

    delete[] m_h_T0k;
    delete[] m_h_dTk;
    delete[] m_h_D0k;
    delete[] m_h_dDk;
    delete[] m_h_Dk;
    delete[] m_h_Tk;
    delete[] m_h_Mu0k;
    delete[] m_h_dMuk;
    delete[] m_h_dAlphak;
    delete[] m_h_alphak;
    delete[] m_h_densRestk;
    delete[] m_h_T;
    delete[] m_h_D;
    delete[] m_h_partPhases;
  }
}
/*****************************************************************************/
/*****************************************************************************/
void MSPHSystem::update()
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

    double timeStep = calculateTimeStep();
    dt = timeStep;

    // integrate
    integrate();

    // Interpolate velocities
    interpolateVelocities();

    // Collision
    collide();

/*****************************************************************************/
/*****************************************************************************/
   // Non-constant fluid diffusion
   for(int i=0;i<phases.size();i++){
   	diffusionEvolution(m_d_dDk, m_d_D0k, m_d_dTk, EA, m_gasStiffness, particles.size(), maxPhases, i);
	temperatureEvolution(m_d_dTk, m_d_Dk, m_dMass, m_d_Tk, m_density, m_dPos[1], m_interactionRadius, voisines, m_d_partPhases, particles.size(), maxPhases, i);
	viscosityEvolution(m_d_dMuk,m_d_dTk, m_d_Tk, m_d_dDk, m_d_Dk, m_gasStiffness, m_interactionRadius,m_d_densRestk , particles.size(),maxPhases,i);
   } 
   integrate_D_T_Mu(m_d_D, m_d_Dk, m_d_T, m_d_Tk, m_viscosity, m_d_dDk, m_d_dTk, m_d_dMuk, m_d_alphak, tempMin, tempMax, muMin, muMax, dt, particles.size(), phases.size(),maxPhases);

   // Concentration evolution
   for(int i=0;i<phases.size();i++)
	concentrationEvolution(m_d_dAlphak, m_d_alphak, m_d_Dk, m_dPos[1], m_dMass, m_density, m_interactionRadius, m_d_partPhases, voisines, particles.size(), maxPhases, i);
   integrateConcentration(m_d_dAlphak, m_d_alphak, dt, particles.size(), phases.size(), maxPhases);

/*****************************************************************************/
/*****************************************************************************/
    copyArrayFromDevice(m_h_alphak,m_d_alphak,0,sizeof(double)*particles.size()*maxPhases);
    for(unsigned int i=0;i<particles.size();i++){
        m_hColors[i*4] = 0;
    	m_hColors[(i*4)+1] = 0;
    	m_hColors[(i*4)+2] = 0;
	for(unsigned int j=0;j<phases.size();j++){
		//printf("Color phase:%d: %f %f %f\n",j,phases[j]->getColor().x(),phases[j]->getColor().y(),phases[j]->getColor().z());
		m_hColors[(i*4)] = m_hColors[(i*4)] + m_h_alphak[i*maxPhases+j]*phases[j]->getColor().x(); 
		m_hColors[(i*4)+1] = m_hColors[(i*4)+1] + m_h_alphak[i*maxPhases+j]*phases[j]->getColor().y();
		m_hColors[(i*4)+2] = m_hColors[(i*4)+2] + m_h_alphak[i*maxPhases+j]*phases[j]->getColor().z();
		if(m_h_alphak[i*maxPhases+j]>1)
			printf("phase:%d conc:%f, result:%f %f %f\n",j,m_h_alphak[i*maxPhases+j],m_hColors[i*4],m_hColors[(i*4)+1],m_hColors[(i*4)+2]);
		if(m_h_alphak[i*maxPhases+j]<0)
			printf("phase:%d conc:%f, result:%f %f %f\n",j,m_h_alphak[i*maxPhases+j],m_hColors[i*4],m_hColors[(i*4)+1],m_hColors[(i*4)+2]);
	}
	//printf("color:%f %f %f\n",m_hColors[i*4], m_hColors[i*4+1], m_hColors[i*4+2]);
    }
    //copyArrayFromDevice(m_h_D,m_d_D,0,sizeof(double)*particles.size()*3);
   // copyArrayFromDevice(m_h_T,m_d_T,0,sizeof(double)*particles.size());
   // copyArrayFromDevice(m_hViscosity,m_viscosity,0,sizeof(double)*particles.size());
    //copyArrayFromDevice(m_h_Mu,m_d_Mu,0,sizeof(double)*particles.size()*3);

}
/*****************************************************************************/
/*****************************************************************************/
void MSPHSystem::displayParticlesByField(uint field)
{
	
	SPHSystem::displayParticlesByField(field);

	// DISPLAY BY VISCOSITY FIELD
        if(field==2){
    		copyArrayFromDevice(m_hViscosity,m_viscosity,0,sizeof(double)*particles.size());
		#pragma omp parallel for
    		for(uint i=0;i<particles.size();i++){
			float hue;
			if(muMin!=muMax) hue = 240 * (muMax - min((float)m_hViscosity[i],muMax)) / (muMax-muMin);
			else hue = 240;
			Vector3 Hsv(hue,1,1);
			Vector3 Rgb = convertHsvToRgb(Hsv);
			m_hColors[i*4] = Rgb.x();
			m_hColors[i*4+1] = Rgb.y();
			m_hColors[i*4+2] = Rgb.z();
    		}
	}
	// Display By Temperature
	if(field == 4){
		copyArrayFromDevice(m_h_T,m_d_T,0,sizeof(double)*particles.size());
		//printf("tempMin:%f tempMax:%f\n",tempMin,tempMax);
		#pragma omp parallel for
		for(uint i=0;i<particles.size();i++){
			float hue;
			if(tempMin!=tempMax) hue = 240 * (tempMax - min((float)m_h_T[i],tempMax)) / (tempMax-tempMin);
			else hue = 240;
			Vector3 Hsv(hue,1,1);
			Vector3 Rgb = convertHsvToRgb(Hsv);
			m_hColors[i*4] = Rgb.x();
			m_hColors[i*4+1] = Rgb.y();
			m_hColors[i*4+2] = Rgb.z();
		}
	}
	// Display By Mass
	if(field == 5){
		copyArrayFromDevice(m_hMass,m_dMass,0,sizeof(double)*particles.size());
		#pragma omp parallel for
		for(uint i=0;i<particles.size();i++){
			float hue;
			if(massMax!=massMin) hue = 240 * (massMax - min((float)m_hMass[i],massMax)) / (massMax-massMin);
			else hue = 240;
			Vector3 Hsv(hue,1,1);
			Vector3 Rgb = convertHsvToRgb(Hsv);
			m_hColors[i*4] = Rgb.x();
			m_hColors[i*4+1] = Rgb.y();
			m_hColors[i*4+2] = Rgb.z();
		}
	}
	// Display by concentration field
	if(field == 6){
		uint indexPhase = 1;
		copyArrayFromDevice(m_h_alphak,m_d_alphak,0,sizeof(double)*particles.size()*maxPhases);
		float alphaMin = 0;
		float alphaMax = 1;
		#pragma omp parallel for
		for(uint i=0;i<particles.size();i++){
			float hue = 240 * (alphaMax - min((float)m_h_alphak[indexPhase*maxPhases + i],alphaMax)) / (alphaMax-alphaMin);
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
void MSPHSystem::_exportData_Mitsuba(const char* filenameDensity, const char* filenameAlbedo, 
				     const char* filenameSigmaS, const char* filenameSigmaT)
{
/*	copyArrayFromDevice(m_hDensity,m_density,0,sizeof(double)*particles.size());

	_exportData(filenameDensity, grid, m_hPos[1],m_hInteractionRadius, m_hDensity, particles.size());
	_exportAlbedo(filenameAlbedo, grid, m_hPos[1],m_hInteractionRadius, m_hAlbedo);
	//_exportAlbedo(filenameSigmaS, grid, m_hPos[1],m_hInteractionRadius, m_hBeta);
	//_exportAlbedo(filenameSigmaT, grid, m_hPos[1],m_hInteractionRadius, m_hSigma);*/
}
/*****************************************************************************/
/*****************************************************************************/
