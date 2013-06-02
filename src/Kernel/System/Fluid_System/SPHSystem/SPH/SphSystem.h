#ifndef __SPH_SYSTEM_H__
#define __SPH_SYSTEM_H__

#include <SimulationData_SPHSystem.h>
#include <SphParticle.h>
#include <System.h>
#include <MeshCollision.h>
#include <SphSystem.cuh>
#include <UniformGrid.cuh>

using namespace std;

// CUDA SPH particles system runs on the GPU

class SPHSystem : public System
{
public:

    SPHSystem();
    virtual ~SPHSystem();

    virtual void init(vector<Particle*> particles);
    virtual void displayParticles(ParticleDisplay mode, Vector3 color);
    virtual void emitParticles();
    virtual void update();

    inline UniformGrid* getGrid(){ return grid; }
    inline bool        getGridCreated(){ return gridCreated; }

    void  _exportData_Mitsuba(const char* filename);

public: // methods

    virtual void  init();
    virtual void _initialize(int begin, int numParticles);
    virtual void _finalize();
    
    void evaluateDensitiesForces();
    void evaluateForcesExt();
    void integrate();
    void collide();
    void interpolateVelocities();

    /*****************************************************************************************/
    void evaluateIso(double* pos, uint nbCellsX, uint nbCellsY, uint nbCellsZ, double scale);
    /*****************************************************************************************/
    void computeNormales_Vertex(double* pos, double* normales, float scale, uint nbV);
    /*****************************************************************************************/

    double* m_interactionRadius;
    double* m_density;
    double* m_restDensity;
    double* m_pressure;
    double* m_gasStiffness;
    double* m_threshold;
    double* m_surfaceTension;
    double* m_viscosity;
    double* m_wj;

    double* m_fPressure;
    double* m_fViscosity;
    double* m_fSurface;
    double* m_normales;
    
    double* m_dVelInterAv;
    double* m_dVelInterAp;
    double* m_hVelInterAv;
    double* m_hVelInterAp;
   
    partVoisine voisines;

    double* m_hMass;
    double* m_hParticleRadius;
    double* m_hInteractionRadius;
    double* m_hDensity;
    double* m_hRestDensity;
    double* m_hPressure;
    double* m_hGasStiffness;
    double* m_hThreshold;
    double* m_hSurfaceTension;
    double* m_hViscosity;
    double* m_hNormales;

    UniformGrid *grid;
    bool gridCreated;

    float m_dt;
};

#endif //
