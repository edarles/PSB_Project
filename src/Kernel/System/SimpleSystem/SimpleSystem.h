#ifndef __SIMPLE_SYSTEM_H__
#define __SIMPLE_SYSTEM_H__

#include <SimulationData_SimpleSystem.h>
#include <Particle.h>
#include <System.h>

using namespace std;

class SimpleSystem : public System
{
public:

    SimpleSystem();
    virtual ~SimpleSystem();

    void init();
    void init(vector<Particle*> particles);
    void emitParticles();

    void update();
    void collide();

protected: // methods
    
    void evaluateForcesExt();
    void integrate();

    void _initialize(int begin, int numParticle);
    void _finalize();

};

#endif //
