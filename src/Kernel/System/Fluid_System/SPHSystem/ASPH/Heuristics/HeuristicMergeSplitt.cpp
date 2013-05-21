#include <HeuristicMergeSplitt.h>

/**************************************************************************************/
/**************************************************************************************/
HeuristicMergeSplitt::HeuristicMergeSplitt(unsigned int nbBodies)
{
	this->nbBodies = nbBodies;
}
/**************************************************************************************/
HeuristicMergeSplitt::HeuristicMergeSplitt(const HeuristicMergeSplitt& H)
{
	freeArray(m_dRes);
	free(m_hRes);
	this->nbBodies = H.nbBodies;
	unsigned int memSize = sizeof(bool)*nbBodies;
    	allocateArray((void**)&m_dRes, memSize);
	m_hRes = new bool[nbBodies];
	memset(m_hRres,0,memSize);
	copyArrayToDevice(H.m_hRes, m_dRres, 0, memSize);
	copyArrayFromDevice(m_dRes, m_hRes, 0, memSize);
}
/**************************************************************************************/
HeuristicMergeSplitt::~HeuristicMergeSplitt()
{
	freeArray(m_dRes);
	free(m_hRes);
}
/**************************************************************************************/
/**************************************************************************************/
bool* HeuristicMergeSplitt::getResult()
{
	return m_hRes;
}
/**************************************************************************************/
unsigned int HeuristicMergeSplitt::getNbBodies()
{
	return nbBodies;
}
/**************************************************************************************/
/**************************************************************************************/
void HeuristicMergeSplitt::changeNbBodies()
{
	if(nbBodies!=this->nbBodies)
	{
		unsigned int memSize = sizeof(bool)*nbBodies;
		bool tempRes = new bool[nbBodies];
		memset(tempRes,0,memSize);
		for(unsigned int i=0;i<nbBodies;i++)
			tempRes[i] = false;
		copyArrayFromDevice(tempRes, m_dRes, 0, sizeof(bool)*this->nbBodies);
		this->nbBodies = nbBodies;
		freeArray(m_dRes);
		free(m_hRes);
		init();
		copyArrayToDevice(tempRes, m_dRes, 0, memSize);
		copyArrayFromDevice(m_dRres, m_hRes, 0, memSize);
}
/**************************************************************************************/
/**************************************************************************************/
HeuristicMergeSplitt::init()
{
	unsigned int memSize = sizeof(bool)*nbBodies;
    	allocateArray((void**)&m_dRes, memSize);
	m_hRes = new bool[nbBodies];
	memset(m_hRres,0,memSize);
	for(unsigned int i=0;i<nbBodies;i++)
		m_hRes[i] = false;
	copyArrayToDevice(m_dRes, m_hRes, 0, memSize);
}
/**************************************************************************************/
/**************************************************************************************/
