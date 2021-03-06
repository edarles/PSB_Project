#include <AsphSystem.h>
#include <SPHSystem.h>
#include <SPHSystem.cuh>

/*****************************************************************************/
/*****************************************************************************/
ASPHSystem::ASPHSystem():SPHSystem()
{
  heuristics.clear();
}
/*****************************************************************************/
ASPHSystem::ASPHSystem(Vector3 origin, int r, float spacing):SPHSystem()
{
  heuristics.clear();
}
/*****************************************************************************/
ASPHSystem::~ASPHSystem()
{
  for(unsigned int i=0;i<heuristics.size();i++)
	delete(heuristics[i]);
  heuristics.clear();
}

/*****************************************************************************/
/*****************************************************************************/
void ASPHSystem::update()
{
    // emit particles
    emitParticles();

    copyArrayToDevice(m_dPos[0], m_hPos[1], 0, sizeof(float)*3*particles.size());
    copyArrayToDevice(m_dVel[0], m_hVel[1], 0, sizeof(float)*3*particles.size());
    for(unsigned int i=0;i<particles.size();i++){
	m_hForce[(i*3)] = 0; m_hForce[(i*3)+1] = 0; m_hForce[(i*3)+2] = 0; 
    }
    copyArrayToDevice(FExt->m_F, m_hForce, 0, sizeof(float)*3*particles.size());

    // check if particles can must be merged or splitted
    checkMergedSplitted();

    // evaluate densities, normales and pressure, viscosity and surface tension forces
    evaluateDensitiesForces();

    // evaluate forces
    evaluateForcesExt();

    // integrate
    integrate();

    // Collision
    collide();

    for(unsigned int i=0;i<particles.size();i++){
	Particle* p = particles[i];
	p->setNewPos(Vector3(m_hPos[1][i*3],m_hPos[1][i*3+1],m_hPos[1][i*3+2]));
	p->setNewVel(Vector3(m_hVel[1][i*3],m_hVel[1][i*3+1],m_hVel[1][i*3+2]));
    }
}
/*************************************************************************************/
/*************************************************************************************/
void ASPHSystem::checkMergedSplitted()
{
	resMerged = heuristics.evaluateFunction(m_dPos[1],m_dVel[1],FExt->m_F);
}
