#ifndef __PCI_SPH_SYSTEM_H__
#define __PCI_SPH_SYSTEM_H__

#include <PciSphParticle.h>
#include <SphSystem.h>

using namespace std;

class PCI_SPHSystem : public SPHSystem
{
public:

    PCI_SPHSystem();
    virtual ~PCI_SPHSystem();

    void init(vector<Particle*> particles);
    void emitParticles();
    void update();

protected: // methods
    
    void evaluateChangesFields();
  
    void _initialize(int begin, int numParticles);

    double* m_restore_dPos[2];
    double* m_restore_velInterAv;
    double* m_restore_velInterAp;

    double* m_hRestore_posAv;
    double* m_hRestore_posAp;
    double* m_hRestore_velInterAv;
    double* m_hRestore_velInterAp;

    double* m_densityError;
    double  errorThreshold;

    double* m_temperatures;
    double* m_hTemperatures;

    double* m_kernelParticles;
    double* m_hKernelParticles;

    double* m_DT;
    double* m_DVisc;
    double* m_DRestDens;
    double* m_DMass;

    int nbIt;
};

#endif //
