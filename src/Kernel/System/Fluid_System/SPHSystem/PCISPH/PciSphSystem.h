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
    virtual double calculateTimeStep();

protected: // methods
    
    void _initialize(int begin, int numParticles);

    double* m_dPosPredict[2];
    double* m_dVelPredict[2];
    double* m_dVelInterPredict[2];
    double* m_densityError;
    double* m_densityPredict;

  
    double* m_hDensityCorrected;
    double* m_hErrorDensity;

    double  errorThreshold;
};

#endif //
