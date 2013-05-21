#include <SimulationDatas.h>

/****************************************************************************/
/****************************************************************************/
SimulationDatas::SimulationDatas()
{
	datas.clear();
}
/****************************************************************************/
SimulationDatas::~SimulationDatas()
{
	for(unsigned int i=0;i<datas.size();i++)
		delete(datas[i]);
	datas.clear();
}
/****************************************************************************/
/****************************************************************************/
SimulationData* SimulationDatas::getData(unsigned int i)
{
	assert(i<datas.size());
	return datas[i];
}
/****************************************************************************/
vector<SimulationData*> SimulationDatas::getDatas()
{
	return datas;
}
/****************************************************************************/
unsigned int SimulationDatas::getNbDatas()
{
	return datas.size();
}
/****************************************************************************/
/****************************************************************************/
void SimulationDatas::setData(unsigned int i, SimulationData* S)
{
	assert(i<datas.size());
	datas[i] = S;
}
/****************************************************************************/
void SimulationDatas::setData(SimulationData* S)
{
	datas.push_back(S);
}
/****************************************************************************/
/****************************************************************************/
