#ifndef SIMULATION_DATA_CUDA_SYSTEM
#define SIMULATION_DATA_CUDA_SYSTEM

#include <Vector3.h>
#include <SimulationData_SimpleSystem.h>

/**********************************************************************************/
/**********************************************************************************/
class SimulationData_CudaSystem : public SimulationData_SimpleSystem {

	public:

/**********************************************************************************/
/**********************************************************************************/
		SimulationData_CudaSystem();
		SimulationData_CudaSystem(float particleRadius, float mass, Vector3 color, float interactionRadius,
         		   		  float spring, float damping, float shear, float attraction);

		SimulationData_CudaSystem(const SimulationData_CudaSystem& S);

		~SimulationData_CudaSystem();

/**********************************************************************************/
/**********************************************************************************/
		float getInteractionRadius();
		float getSpring();
		float getDamping();
		float getShear();
		float getAttraction();

/**********************************************************************************/
/**********************************************************************************/
		void  setInteractionRadius(float);
		void  setSpring(float);
		void  setDamping(float);
		void  setShear(float);
		void  setAttraction(float);

/**********************************************************************************/
/**********************************************************************************/
		bool loadConfiguration(const char* filename);
		bool saveConfiguration(const char* filename);

	private:
/**********************************************************************************/
/**********************************************************************************/
		float interactionRadius;
		float spring;
		float damping;
		float shear;
		float attraction;
/**********************************************************************************/
/**********************************************************************************/
};
/**********************************************************************************/
/**********************************************************************************/
#endif
