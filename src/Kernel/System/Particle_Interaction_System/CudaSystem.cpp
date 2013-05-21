#include <GLee.h>
#include <CudaParticle.h>
#include <CudaSystem.h>
#include <System.cuh>
#include <CudaSystem.cuh>
#include <ForceExt.cuh>
#include <typeinfo>

/*****************************************************************************/
/*****************************************************************************/
CudaSystem::CudaSystem():System()
{
    m_solverIterations = 1;
    dt = 0.01;
    m_dPos[0] = m_dPos[1] = 0;
    m_dVel[0] = m_dVel[1] = 0;
    m_hPos[0] = m_hPos[1] = 0;
    m_hVel[0] = m_hVel[1] = 0;
    m_hForce = 0;
    m_dMass = 0;
    m_dParticleRadius = 0;
    m_dInteractionRadius = 0;
    m_dSpring = 0;
    m_dDamping = 0;
    m_dShear = 0;
    m_dAttraction = 0;

    gridCreated = false;

    init();
}
/*****************************************************************************/
CudaSystem::~CudaSystem()
{
   for(unsigned int i=0;i<particles.size();i++)
	delete(particles[i]);
   particles.clear();
   _finalize();
}
/*****************************************************************************/
void CudaSystem::init()
{
    unsigned int maxParticles = 65536;
    unsigned int memSize = sizeof(double)*3*maxParticles;
    
    allocateArray((void**)&m_dVel[0], memSize);
    allocateArray((void**)&m_dVel[1], memSize);
    allocateArray((void**)&m_dPos[0], memSize);
    allocateArray((void**)&m_dPos[1], memSize);
    allocateArray((void**)&m_dMass, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dParticleRadius, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dInteractionRadius, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dSpring, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dDamping, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dShear, sizeof(double)*maxParticles);
    allocateArray((void**)&m_dAttraction, sizeof(double)*maxParticles);

    m_hPos[0] = new double[maxParticles*3];
    m_hVel[0] = new double[maxParticles*3];
    m_hPos[1] = new double[maxParticles*3];
    m_hVel[1] = new double[maxParticles*3];
    m_hForce = new double[maxParticles*3];
    m_hColors = new double[maxParticles*4];   
    m_hMass = new double[maxParticles];
    m_hParticleRadius = new double[maxParticles];
    m_hInteractionRadius = new double[maxParticles];
    m_hSpring = new double[maxParticles];
    m_hDamping = new double[maxParticles];
    m_hShear = new double[maxParticles];
    m_hAttraction = new double[maxParticles];

    allocateArray((void**)&voisines.nbVoisines,maxParticles*sizeof(int));
    allocateArray((void**)&voisines.listeVoisine,maxParticles*200*sizeof(int));

    FExt->_initialize(maxParticles);
    gridCreated = false;

}
/*****************************************************************************/
void CudaSystem::init(vector<Particle*> particles)
{
      this->particles.clear();
      for(unsigned int i=0;i<particles.size();i++)
		this->particles.push_back(particles[i]);
      _initialize(0,particles.size());
}
/*****************************************************************************/
void CudaSystem::_initialize(int begin, int numParticles)
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
	 CudaParticle* p   = (CudaParticle*) particles[i];
	
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

         m_hInteractionRadius[i] = p->getInteractionRadius();
         m_hSpring[i] = p->getSpring();
         m_hDamping[i] = p->getDamping();
         m_hShear[i] = p->getShear();
         m_hAttraction[i] = p->getAttraction();

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
    copyArrayToDevice(m_dInteractionRadius, m_hInteractionRadius, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_dSpring, m_hSpring, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_dDamping, m_hDamping, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_dShear, m_hShear, 0, sizeof(double)*numParticles);
    copyArrayToDevice(m_dAttraction, m_hAttraction, 0, sizeof(double)*numParticles);
   
    if(gridCreated==false){
	CudaParticle* p   = (CudaParticle*) particles[0];
	grid = (UniformGrid*)malloc(sizeof(UniformGrid));
	createGrid(0,0,0,1.5,1.5,1.5,p->getInteractionRadius(),grid);
        gridCreated = true;
    }
  }
}
/*****************************************************************************/
void CudaSystem::_finalize()
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
    delete[] m_hInteractionRadius;
    delete[] m_hSpring;
    delete[] m_hDamping;
    delete[] m_hShear;
    delete[] m_hAttraction;

    freeArray(m_dPos[0]);
    freeArray(m_dPos[1]);
    freeArray(m_dVel[0]);
    freeArray(m_dVel[1]);
    freeArray(m_dMass);
    freeArray(m_dParticleRadius);
    freeArray(m_dInteractionRadius);
    freeArray(m_dSpring);
    freeArray(m_dDamping);
    freeArray(m_dShear);
    freeArray(m_dAttraction);   
    freeArray(voisines.listeVoisine);
    freeArray(voisines.nbVoisines);   

    FExt->_finalize();

    _finalizeGrid(grid);
  }
}
/*****************************************************************************/
/*****************************************************************************/
void CudaSystem::emitParticles()
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
void CudaSystem::update()
{
    // emit particles
    emitParticles();

    reinitCells(grid);
    storeParticles(grid,m_dPos[1],particles.size());
    searchNeighbooring(m_dPos[1],m_dInteractionRadius, &(*grid), 1, voisines, particles.size());

    // evaluate forces
    evaluateForcesExt();

    // integrate
    integrate();

    // Collision
    collide();
}
/*****************************************************************************/
void CudaSystem::evaluateForcesExt()
{
   for(unsigned int i=0;i<particles.size();i++){
	m_hForce[(i*3)] = 0; m_hForce[(i*3)+1] = 0; m_hForce[(i*3)+2] = 0; 
   }
   copyArrayToDevice(FExt->m_F, m_hForce, 0, sizeof(double)*3*particles.size());
   for(unsigned int i=0;i<FExt->getNbForces();i++){
		if(typeid(*(FExt->getForce(i))) == typeid(ForceExt_Constante)){
			ForceExt_Constante* FC = (ForceExt_Constante*) FExt->getForce(i);
			evaluate_ForceExt_Constante(particles.size(), FExt->m_F, m_dMass, FC); 
		}
		if(typeid(*(FExt->getForce(i))) == typeid(ForceExt_Trochoide)){
			ForceExt_Trochoide *T = (ForceExt_Trochoide*) FExt->getForce(i);
			evaluate_ForceExt_Trochoide(particles.size(), m_hPos[1], FExt->m_F, m_dMass, T);	
			T->setTime(T->getTime()+1); 
		}
   }
   interactionSystem(m_dPos[0], m_dVel[0], FExt->m_F, m_dInteractionRadius, m_dSpring, m_dDamping,
		     m_dShear, m_dAttraction, voisines, particles.size());
}
/*****************************************************************************/
void CudaSystem::integrate()
{
    integrateSystem(m_dPos[0], m_dPos[1],m_dVel[0], m_dVel[1], FExt->m_F, m_dMass, dt, 0, particles.size());
}
/*****************************************************************************/

void CudaSystem::collide()
{
	collision->collide(m_dPos[0],m_dPos[1],m_dVel[0], m_dVel[1], 
			   particles[0]->getParticleRadius(), dt, particles.size());

    	copyArrayFromDevice(m_hPos[1], m_dPos[1], 0, sizeof(double)*3*particles.size());
}
