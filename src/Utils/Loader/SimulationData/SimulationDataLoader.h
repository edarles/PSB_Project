#ifndef _SIMULATION_DATA_LOADER_
#define _SIMULATION_DATA_LOADER_

#include <SimulationData.h>

using namespace std;

namespace Utils {

class SimulationDataLoader
{
	public:
		SimulationDataLoader();
		~SimulationDataLoader();

		virtual SimulationData *load(const char *filename) = 0;

};

}
#endif 
