#include <GLee.h>
#include <PciSphSystem.h>
#include <PciSphSystem.cuh>

/*****************************************************************************/
/*****************************************************************************/
PCI_SPHSystem::PCI_SPHSystem():SPHSystem()
{
	unsigned int maxParticles = 65536;
   	unsigned int memSize = sizeof(double)*3*maxParticles;
    
        allocateArray((void**)&m_restore_dPos[0], memSize);
        allocateArray((void**)&m_restore_dPos[1], memSize);
        allocateArray((void**)&m_restore_velInterAv, memSize);
        allocateArray((void**)&m_restore_velInterAp, memSize);
	allocateArray((void**)&m_densityError, memSize);

        m_hRestore_posAv = new double[maxParticles];
        m_hRestore_posAp = new double[maxParticles];
        m_hRestore_velInterAv = new double[maxParticles];
        m_hRestore_velInterAp = new double[maxParticles];

        errorThreshold = 10;
}

/*****************************************************************************/
PCI_SPHSystem::~PCI_SPHSystem()
{
	freeArray(m_restore_dPos[0]);
        freeArray(m_restore_dPos[1]);
        freeArray(m_restore_velInterAv);
        freeArray(m_restore_velInterAp);
	freeArray(m_densityError);

	delete[] m_hRestore_posAv;
	delete[] m_hRestore_posAp;
        delete[] m_hRestore_velInterAv;
        delete[] m_hRestore_velInterAp;
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

	copyArrayFromDevice(m_hRestore_posAv, m_restore_dPos[0], 0, sizeof(double)*3*begin);
	copyArrayFromDevice(m_hRestore_posAp, m_restore_dPos[1], 0, sizeof(double)*3*begin);
	copyArrayFromDevice(m_hRestore_velInterAv, m_restore_velInterAv, 0, sizeof(double)*3*begin);
	copyArrayFromDevice(m_hRestore_velInterAp,m_restore_velInterAp, 0, sizeof(double)*3*begin);
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

	 m_hRestore_posAv[i*3]   = p->getOldPos().x();
	 m_hRestore_posAv[i*3+1] = p->getOldPos().y();
	 m_hRestore_posAv[i*3+2] = p->getOldPos().z();

	 m_hRestore_posAp[i*3]   = p->getNewPos().x();
	 m_hRestore_posAp[i*3+1] = p->getNewPos().y();
	 m_hRestore_posAp[i*3+2] = p->getNewPos().z();

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
    copyArrayToDevice(m_restore_dPos[0], m_hRestore_posAv, 0, sizeof(double)*3*particles.size());
    copyArrayToDevice(m_restore_dPos[1], m_hRestore_posAp, 0, sizeof(double)*3*particles.size());
    copyArrayToDevice(m_restore_velInterAv, m_hRestore_velInterAv, 0, sizeof(double)*3*particles.size());
    copyArrayToDevice(m_restore_velInterAp, m_hRestore_velInterAp, 0, sizeof(double)*3*particles.size());

    copyArrayToDevice(m_dMass, m_hMass, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_dParticleRadius, m_hParticleRadius, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_interactionRadius, m_hInteractionRadius, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_restDensity, m_hRestDensity, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_gasStiffness, m_hGasStiffness, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_threshold, m_hThreshold, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_viscosity, m_hViscosity, 0, sizeof(double)*numParticles);  
  
    if(gridCreated==false){
	PCI_SPHParticle* p   = (PCI_SPHParticle*) particles[0];
	grid = (UniformGrid*)malloc(sizeof(UniformGrid));
	createGrid(0,0,0,1.0,1.0,1.0,p->getInteractionRadius(),grid);
        gridCreated = true;
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

    // evaluate forces
 /*   FExt->init(particles.size());
    for(unsigned int i=0;i<FExt->getNbForces();i++){
		if(typeid(*(FExt->getForce(i))) == typeid(ForceExt_Constante)){
			ForceExt_Constante* FC = (ForceExt_Constante*) FExt->getForce(i);
			evaluate_ForceExt_Constante(particles.size(),FExt->m_F, m_density, FC); 
		}
		if(typeid(*(FExt->getForce(i))) == typeid(ForceExt_Trochoide)){
			ForceExt_Trochoide *T = (ForceExt_Trochoide*) FExt->getForce(i);
			evaluate_ForceExt_Trochoide(particles.size(), m_hPos[1], FExt->m_F, m_density, T);	
			T->setTime(T->getTime()+1); 
		}
   }*/
   float dt1 = 0.005;
   
   pci_SPH_update(m_dPos[0],m_dPos[1],m_dVel[1],m_dVelInterAv,m_dVelInterAp,
		   m_restore_dPos[0],m_restore_dPos[1],m_restore_velInterAv,m_restore_velInterAp,
		   m_dMass, m_interactionRadius, m_density, m_restDensity, m_densityError, m_pressure, 
		   m_gasStiffness, m_viscosity, m_threshold, m_surfaceTension, errorThreshold, voisines,
		   m_normales, m_fViscosity, m_fSurface, m_fPressure, FExt->m_F, dt1, particles.size());

    // Collision
    collide();

    // Interpolate Velocities
    interpolateVelocities();

    //GPU to GPU positions transfert 
    copyArrayDeviceToDevice(m_restore_dPos[0], m_dPos[0],0, sizeof(double)*3*particles.size());
    copyArrayDeviceToDevice(m_restore_dPos[1], m_dPos[1],0, sizeof(double)*3*particles.size());
    copyArrayDeviceToDevice(m_restore_velInterAv, m_dVelInterAv, 0, sizeof(double)*3*particles.size());
    copyArrayDeviceToDevice(m_restore_velInterAp, m_dVelInterAp, 0, sizeof(double)*3*particles.size());
}
/*****************************************************************************/
/*****************************************************************************/
