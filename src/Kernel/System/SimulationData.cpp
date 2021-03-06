#include "SimulationData.h"

/***************************************************************************/
/***************************************************************************/
SimulationData::SimulationData()
{
	particleRadius = 0;
	particleMass = 0;
	color = Vector3(1,1,1);
}
/***************************************************************************/
SimulationData::SimulationData(float particleRadius, float particleMass, Vector3 color)
{
	this->particleRadius = particleRadius;
	this->particleMass = particleMass;
	this->color.setXYZ(color.x(),color.y(),color.z());
}
/***************************************************************************/
SimulationData::SimulationData(const SimulationData& S)
{
	this->particleRadius = S.particleRadius;
	this->particleMass = S.particleMass;
	this->color = S.color;
}
/***************************************************************************/
SimulationData::~SimulationData()
{
}
/***************************************************************************/
/***************************************************************************/
float SimulationData::getParticleRadius()
{
	return particleRadius;
}
/***************************************************************************/
float SimulationData::getParticleMass()
{
	return particleMass;
}
/***************************************************************************/
Vector3 SimulationData::getColor()
{
	return color;
}
/***************************************************************************/
/***************************************************************************/
void SimulationData::setParticleRadius(float pr)
{
	this->particleRadius = pr;
}
/***************************************************************************/
void SimulationData::setParticleMass(float m)
{
	this->particleMass = m;
}
/***************************************************************************/
void SimulationData::setColor(Vector3 color)
{
	this->color.setXYZ(color.x(),color.y(),color.z());
}
/*********************************************************************/
/***************************************************************************/
