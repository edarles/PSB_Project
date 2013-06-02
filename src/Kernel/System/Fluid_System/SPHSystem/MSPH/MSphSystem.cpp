#include <MSphSystem.h>
#include <MSphSystem.cuh>
/*****************************************************************************/
/*****************************************************************************/
MSPHSystem::MSPHSystem():SPHSystem()
{
  init();
}
/*****************************************************************************/
MSPHSystem::~MSPHSystem()
{
}
/*****************************************************************************/
/*****************************************************************************/
void MSPHSystem::init()
{
    unsigned int maxParticles = 65536;
    SPHSystem::init();

    allocateArray((void**)&m_dTemp, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dCij, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dDeltaTemp, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dDeltaM, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dDeltaRho0, sizeof(double)*maxParticles);

    m_hTemp = new double[maxParticles];

    tempMoy = 0;
    MMoy = 0;
    Rho0Moy = 0;
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

	copyArrayFromDevice(m_hTemp, m_dTemp, 0, sizeof(double)*begin);
    }
    printf("begin:%d numParticles:%d\n",begin, numParticles);
   
    double tTempMoy = 0;
    double tMMoy = 0;
    double tRho0Moy = 0;

    for(int i=begin;i<numParticles;i++){
	 MSPHParticle* p   = (MSPHParticle*) particles[i];
	
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
	 m_hColors[(i*4)+3] = 0.8; 

	 m_hTemp[i] = p->getTemperature();

         tTempMoy += p->getTemperature();
         tMMoy += p->getMass();
         tRho0Moy += p->getRestDensity();
   }
   tempMoy = (tempMoy*begin + tTempMoy)/particles.size();
   MMoy =  (MMoy*begin + tMMoy)/particles.size();
   Rho0Moy = (Rho0Moy*begin + tRho0Moy)/particles.size();;

    printf("TempMoy:%f MassMoy:%f Rho0Moy:%f\n",tempMoy,MMoy,Rho0Moy);

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

    copyArrayToDevice(m_dTemp, m_hTemp, 0, sizeof(double)*numParticles);

    if(gridCreated==false){
	MSPHParticle* p   = (MSPHParticle*) particles[0];
	grid = (UniformGrid*)malloc(sizeof(UniformGrid));
	createGrid(0,0,0,1.0,1.0,1.0,p->getInteractionRadius(),grid);
        gridCreated = true;
    }
 
  }
}
/*****************************************************************************/
void MSPHSystem::_finalize()
{
  if(particles.size()>0)
  {
    SPHSystem::_finalize();
    freeArray(m_dTemp);
    freeArray(m_dCij);
    freeArray(m_dDeltaTemp);
    freeArray(m_dDeltaM);
    freeArray(m_dDeltaRho0);

      delete[] m_hTemp;
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

    // integrate
    integrate();

    // Collision
    collide();

    // Interpolate velocities
    interpolateVelocities();

/*****************************************************************************/
/*****************************************************************************/
    //Temperature, mass and rest densities evolution
    evaluate_T_Rho0_Mass (m_dPos[1], m_dMass, m_interactionRadius, m_density, 
			  m_restDensity, m_dTemp, m_dCij, voisines,
		          m_dDeltaTemp, m_dDeltaM, m_dDeltaRho0, particles.size());
    //Integrate changes
    integrate_T_Rho0_Mass(m_dTemp, m_restDensity, m_dMass, tempMoy, MMoy, Rho0Moy, 
			  m_dDeltaTemp, m_dDeltaM, m_dDeltaRho0, dt, particles.size());
/*****************************************************************************/
/*****************************************************************************/
    copyArrayFromDevice(m_hMass,m_dMass,0,sizeof(double)*particles.size());
    copyArrayFromDevice(m_hTemp,m_dTemp,0,sizeof(double)*particles.size());
   
    for(uint i=0;i<particles.size();i++){
	MSPHParticle *p = (MSPHParticle*)particles[i];
	//if(m_hMass[i]!=particles[i]->getMass()){
		// low temperature -> blue
		if(m_hTemp[i]>=20 && m_hTemp[i]<=25){
			m_hColors[i*4] = 0;
			m_hColors[i*4+1] = 0;
			m_hColors[i*4+2] = 1.0;
		}
		// medium low temperature -> green
		if(m_hTemp[i]>=25 && m_hTemp[i]<=30){
			m_hColors[i*4] = 0;
			m_hColors[i*4+1] =  1.0;
			m_hColors[i*4+2] = 0;
		}
		// medium high temperature -> yellow
		if(m_hTemp[i]>=30 && m_hTemp[i]<=35){
			m_hColors[i*4] = 1.0;
			m_hColors[i*4+1] =  1.0;
			m_hColors[i*4+2] = 0;
		}
		// medium high temperature -> red
		if(m_hTemp[i]>=35 && m_hTemp[i]<=40){
			m_hColors[i*4] = 1.0;
			m_hColors[i*4+1] = 0;
			m_hColors[i*4+2] = 0;
		}
/*
		if(particles[i]->getColor().x()==1 && particles[i]->getColor().y()==1 && particles[i]->getColor().z()==1){
			m_hColors[i*4+1] = 1-min(particles[i]->getMass()+fabs(particles[i]->getMass()-m_hMass[i])/particles[i]->getMass(),1.0);
			m_hColors[i*4+2] = 1-min(particles[i]->getMass()+fabs(particles[i]->getMass()-m_hMass[i])/particles[i]->getMass(),1.0);
		}
		if(particles[i]->getColor().x()==1 && particles[i]->getColor().y()==0 && particles[i]->getColor().z()==0){
			m_hColors[i*4+1] = min(particles[i]->getMass()+fabs(particles[i]->getMass()-m_hMass[i])/particles[i]->getMass(),1.0);
			m_hColors[i*4+2] = min(particles[i]->getMass()+fabs(particles[i]->getMass()-m_hMass[i])/particles[i]->getMass(),1.0);
		}*/
	//}
    }
   
}
/*****************************************************************************/
/*****************************************************************************/
