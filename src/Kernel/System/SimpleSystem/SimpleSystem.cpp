#include <GLee.h>
#include <SimpleSystem.h>
#include <System.cuh>

/*****************************************************************************/
/*****************************************************************************/
SimpleSystem::SimpleSystem():System()
{
    m_solverIterations = 1;
    m_dPos[0] = m_dPos[1] = 0;
    m_dVel[0] = m_dVel[1] = 0;
    m_hPos[0] = m_hPos[1] = 0;
    m_hVel[0] = m_hVel[1] = 0;
    m_dMass = 0;
    m_dParticleRadius = 0;

    init();
}

/*****************************************************************************/
SimpleSystem::~SimpleSystem()
{
   _finalize();
   for(unsigned int i=0;i<particles.size();i++)
	delete(particles[i]);
   particles.clear();
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
void SimpleSystem::init(vector<Particle*> result)
{
      this->particles.clear();
      this->particles.insert(particles.end(),result.begin(),result.end());
      _initialize(0,particles.size());
}
/*****************************************************************************/
void SimpleSystem::init()
{
    unsigned int maxParticles = 65536;
    unsigned int memSize = sizeof(double)*3*maxParticles;
    
    allocateArray((void**)&m_dVel[0], memSize);
    allocateArray((void**)&m_dVel[1], memSize);
    allocateArray((void**)&m_dPos[0], memSize);
    allocateArray((void**)&m_dPos[1], memSize);
    allocateArray((void**)&m_dMass, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dParticleRadius, sizeof(double)*maxParticles);

    m_hPos[0] = new double[maxParticles*3];
    m_hVel[0] = new double[maxParticles*3];
    m_hPos[1] = new double[maxParticles*3];
    m_hVel[1] = new double[maxParticles*3];
    m_hForce = new double[maxParticles*3];
    m_hColors = new double[maxParticles*4];   
    m_hMass = new double[maxParticles];
    m_hParticleRadius = new double[maxParticles];
    
    FExt->_initialize(maxParticles);
}

/*****************************************************************************/
void SimpleSystem::_initialize(int begin, int numParticles)
{
  if(numParticles>0){

    if(begin>0){
    	copyArrayFromDevice(m_hPos[0], m_dPos[0], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVel[0], m_dVel[0], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hPos[1], m_dPos[1], 0, sizeof(double)*3*begin);
    	copyArrayFromDevice(m_hVel[1], m_dVel[1], 0, sizeof(double)*3*begin);
    }
    printf("begin:%d numParticles:%d\n",begin, numParticles);
   
    for(int i=begin;i<numParticles;i++){
	 Particle* p   = (Particle*) particles[i];
	
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

	 m_hMass[i] = p->getMass();
	 m_hParticleRadius[i] = p->getParticleRadius();

	 m_hColors[i*4] = p->getColor().x();
	 m_hColors[(i*4)+1] = p->getColor().y();
	 m_hColors[(i*4)+2] = p->getColor().z();
	 m_hColors[(i*4)+3] = 1.0; 

   }
    copyArrayToDevice(m_dPos[0], m_hPos[0], 0, sizeof(double)*3*numParticles);
    copyArrayToDevice(m_dVel[0], m_hVel[0], 0, sizeof(double)*3*numParticles);
    copyArrayToDevice(m_dPos[1], m_hPos[1], 0, sizeof(double)*3*numParticles);
    copyArrayToDevice(m_dVel[1], m_hVel[1], 0, sizeof(double)*3*numParticles);
    copyArrayToDevice(m_dMass, m_hMass, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_dParticleRadius, m_hParticleRadius, 0, sizeof(double)*numParticles);

  }
}
/*****************************************************************************/
void SimpleSystem::_finalize()
{
  if(particles.size()>0)
  {
    delete[] m_hVel[0];
    delete[] m_hPos[0];
    delete[] m_hVel[1];
    delete[] m_hPos[1];
    delete[] m_hForce;
    delete[] m_hColors;
    delete[] m_hMass;
    delete[] m_hParticleRadius;

    freeArray(m_dPos[0]);
    freeArray(m_dPos[1]);
    freeArray(m_dVel[0]);
    freeArray(m_dVel[1]);
    freeArray(m_dMass);
    freeArray(m_dParticleRadius);

    FExt->_finalize();

  }
}
/*****************************************************************************/
/*****************************************************************************/
void SimpleSystem::emitParticles()
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
void SimpleSystem::update()
{
    // emit particles
    emitParticles();
    // evaluate forces
    evaluateForcesExt();
    // integrate
    integrate();
    // Collision
    collide();
}
/*****************************************************************************/
void SimpleSystem::evaluateForcesExt()
{
   FExt->init(particles.size());
   FExt->evaluate(m_dPos[0],FExt->m_F, m_dMass, particles.size());
}
/*****************************************************************************/
void SimpleSystem::integrate()
{  
    integrateSystem(m_dPos[0],m_dPos[1],m_dVel[0], m_dVel[1],FExt->m_F, m_dMass, dt, 0, particles.size());
    copyArrayFromDevice(m_hPos[1], m_dPos[1], 0, sizeof(double)*3*particles.size());
    copyArrayFromDevice(m_hVel[1], m_dVel[1], 0, sizeof(double)*3*particles.size());
}


/*****************************************************************************/
void SimpleSystem::collide()
{
	if(collision->getObjects().size()>0)
	collision->collide(m_dPos[0],m_dPos[1],m_dVel[0],m_dVel[1], 
			   particles[0]->getParticleRadius(), dt, particles.size());
}
