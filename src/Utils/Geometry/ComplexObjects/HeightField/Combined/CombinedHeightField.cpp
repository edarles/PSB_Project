#include <CombinedHeightField.h>
#include <HeightField.cuh>
/*****************************************************************************************************/
/*****************************************************************************************************/
namespace Utils
{
/*****************************************************************************************************/
/*****************************************************************************************************/
Combined_HeightField::Combined_HeightField()
		     :HeightField()
{
	HFields.clear();
}
/*****************************************************************************************************/
Combined_HeightField::Combined_HeightField(Vector3 Min, Vector3 Max, double d_x, double d_z)
		     :HeightField(Min,Max,d_x,d_z)
{
	HFields.clear();
}
/*****************************************************************************************************/
Combined_HeightField::Combined_HeightField(const Combined_HeightField& H):HeightField(H)
{
	this->HFields.clear();
	for(uint i=0;i<H.HFields.size();i++)
		this->HFields.push_back(H.HFields[i]);
}
/*****************************************************************************************************/
Combined_HeightField::~Combined_HeightField()
{
	clearAll();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void Combined_HeightField::addHeightField(HeightField* H)
{
	HFields.push_back(H);
}
/*****************************************************************************************************/
void Combined_HeightField::setHeightField(HeightField* H, uint index)
{
	assert(index<HFields.size());
	HFields[index] = H;
}
/*****************************************************************************************************/
void Combined_HeightField::removeHeightFields(uint index)
{
	assert(index<HFields.size());
	delete(HFields[index]);
}
/*****************************************************************************************************/
void Combined_HeightField::clearAll()
{
	for(uint i=0;i<HFields.size();i++)
		removeHeightFields(i);
	HFields.clear();
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void  Combined_HeightField::create(vector<HeightField*> Hfields)
{
	clearAll();
	for(uint i=0;i<Hfields.size();i++)
		addHeightField(Hfields[i]);
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void  Combined_HeightField::calculateHeight(double* m_pos, uint nbPos)
{
	HeightField_initializeHeight_CUDA(Min.y(),m_pos,nbPos);
	for(uint i=0;i<HFields.size();i++){		
		HFields[i]->calculateHeight(m_pos,nbPos);
	}	
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void  Combined_HeightField::calculateHeight_Normales(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1, uint nbPos)
{
	HeightField_initializeHeight_Normales_CUDA(m_pos,nx0,nx1,nz0,nz1,0.01,0,0.01,nbPos);
	for(uint i=0;i<HFields.size();i++){		
		HFields[i]->calculateHeight_Normales(m_pos,nx0,nx1,nz0,nz1,nbPos);
	}	
}
/*****************************************************************************************************/
/*****************************************************************************************************/
}
/*****************************************************************************************************/
/*****************************************************************************************************/
