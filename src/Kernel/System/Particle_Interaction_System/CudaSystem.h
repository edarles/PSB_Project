#ifndef __CUDA_SYSTEM_H__
#define __CUDA_SYSTEM_H__

#include <SimulationData_CudaSystem.h>
#include <Particle.h>
#include <System.h>
#include <SphSystem.cuh>
#include <UniformGrid.cuh>

using namespace std;

// CUDA Simple particles system runs on the GPU

class CudaSystem : public System
{
public:

    CudaSystem();
    virtual ~CudaSystem();

    void init();
    void init(vector<Particle*> particles);
    void emitParticles();
    void update();
 
protected: // methods
    
    void evaluateForcesExt();
    void integrate();
    void collide();

    void _initialize(int begin, int numParticles);
    void _finalize();

    // GPU STORAGE
    double* m_dInteractionRadius;
    double* m_dSpring;
    double* m_dDamping;
    double* m_dShear;
    double* m_dAttraction;
    UniformGrid *grid;
    partVoisine voisines;

    // CPU STORAGE
    double* m_hInteractionRadius;
    double* m_hSpring;
    double* m_hDamping;
    double* m_hShear;
    double* m_hAttraction;

    bool gridCreated;
};

#endif //
