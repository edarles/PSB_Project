#ifndef _SYSTEM_H__
#define _SYSTEM_H__

#include <cuda.h>
#include <common.cuh>

#include <vector>
#include <SimulationDatas.h>
#include <Particle.h>
#include <Collision.h>
#include <MeshCollision.h>
#include <SphereCollision.h>
#include <BoxCollision.h>
#include <PlaneCollision.h>
#include <ForcesExt.h>
#include <Emitters.h>

#define maxParticles  300000

class System
{
public:

    System();
    virtual ~System();

    enum ParticleDisplay
    {
	POINTS,
	SPHERES,
	SURFACE
    };
    virtual void init(vector<Particle*> particles) = 0;
    virtual void update() = 0;
    virtual void emitParticles() = 0;
    virtual void collide() = 0;

    virtual void displayParticles(ParticleDisplay mode, Vector3 color);
    virtual void displayParticlesByField(uint field);
    virtual void displayCollisions(GLenum, GLenum, Vector3 colorObject, Vector3 colorNormales, bool);
    virtual void displayEmitters(Vector3);

    vector<Particle*> getParticles();
    void	      setParticles(vector<Particle*> particles);

    ObjectCollision* getObjectCollision(unsigned int);
    vector<ObjectCollision*> getObjectsCollision();
    virtual void             addObjectCollision(ObjectCollision*);
    virtual void             removeLastObjectCollision();

    Emitters*	     getEmitters();
    virtual void     addEmitter(Emitter* E);

    ForcesExt*	     getForcesExt();
    virtual void     addForce(ForceExt*);

    int  getNumParticles() const { return particles.size(); }
    void setIterations(int i) { m_solverIterations = i; }

    double getDt();
    void   setDt(double);

   public:

	GLuint m_program;

	// CPU storage data (positions and velocities)
	double* m_hPos[2];
	double* m_hVel[2];
        double* m_hMass;
	double* m_hForce;
	double* m_hColors;
        double* m_hParticleRadius;

	// GPU storage data (positions and velocities)
	double* m_dPos[2];
	double* m_dVel[2];
        double* m_dMass;
	double* m_dParticleRadius;
	double* m_dColor;

	 // CPU Storage particles
        vector<Particle*> particles;

	//SimulationData  *simulationData;
	Collision       *collision;
	Emitters        *emitters;
	ForcesExt       *FExt;

	double		dt;
	int 		t;
	float 		currentTime;
	int 		frameNumber;
	uint            m_solverIterations;

	GLuint _compileProgram(const char *vsource, const char *fsource);
};

#endif // _SYSTEM_H
