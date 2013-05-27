#ifndef _FORCES_EXT_
#define _FORCES_EXT_

#include <ForceExt.h>
#include <ForceExt_Constante.h>
#include <ForceExt_Trochoide.h>

#include <vector>
#include <assert.h>

using namespace std;

class ForcesExt {

	public:
		ForcesExt();
		~ForcesExt();

		ForceExt*    getForce(unsigned int);
		unsigned int getNbForces();

		void setForce(ForceExt*, unsigned int);
		void addForce(ForceExt*);

		//gpu storage accumulator forces
		double* m_F;

		void _initialize(uint nbBodies);
		void _finalize();

		void init(uint nbBodies);

	private:

		vector<ForceExt*> forces;

};

#endif
