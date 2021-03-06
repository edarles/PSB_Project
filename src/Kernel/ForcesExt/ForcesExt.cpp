#include <ForcesExt.h>
#include <ForceExt.cuh>
#include <cuda.h>
#include <typeinfo>

#include <common.cuh>
#include <string.h>
/*****************************************************************************************************/
/*****************************************************************************************************/
ForcesExt::ForcesExt()
{
	forces.clear();
}
/*****************************************************************************************************/
ForcesExt::~ForcesExt()
{
	forces.clear();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
ForceExt* ForcesExt::getForce(unsigned int i)
{
	assert(i<forces.size());
	return forces[i];
}
/*****************************************************************************************************/
unsigned int ForcesExt::getNbForces()
{
	return forces.size();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void ForcesExt::setForce(ForceExt *F, unsigned int i)
{
	assert(i<forces.size());
	forces[i] = F;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void ForcesExt::addForce(ForceExt *F)
{
	forces.push_back(F);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void ForcesExt::_initialize(uint nbBodies)
{
       // initialize forces accumulator buffer and store it in GPU
	unsigned int memSize = sizeof(double) * 3 * nbBodies;
	allocateArray((void**)&m_F, memSize);
}
/*****************************************************************************************************/
void ForcesExt::init(uint nbBodies)
{
	double* F = new double[3*nbBodies];
	for(uint i=0;i<nbBodies;i++){
		F[i*3] = 0;
		F[i*3+1] = 0;
		F[i*3+2] = 0;
	}
	copyArrayToDevice(m_F,F,0,sizeof(double)*3*nbBodies);
	delete[] F;
}
/*****************************************************************************************************/
void ForcesExt::_finalize()
{
	freeArray(m_F); 
}
