#include "Particle.h"

/*********************************************************************************************/
/*********************************************************************************************/
Particle::Particle()
{
	oldPos = Vector3(0,0,0);
	oldVel = Vector3(0,0,0);
	newPos = Vector3(0,0,0);
	newVel = Vector3(0,0,0);
	mass = 0;
	particleRadius = 0;
	color = Vector3(0,0,0);
}
/*********************************************************************************************/
Particle::Particle(Vector3 oldPos, Vector3 oldVel, double mass, double particleRadius, Vector3 color)
{
	this->oldPos = oldPos;
	this->oldVel = oldVel;
	this->newPos = oldPos;
	this->newVel = newVel;
	this->mass = mass;
	this->particleRadius = particleRadius;
	this->color = Vector3(color.x(),color.y(),color.z());
}
/*********************************************************************************************/
Particle::Particle(const Particle& P)
{
	this->oldPos = P.oldPos;
	this->oldVel = P.oldVel;
	this->newPos = P.oldPos;
	this->newVel = P.newVel;
	this->mass = P.mass;
	this->particleRadius = P.particleRadius;
	this->color = P.color;
}
/*********************************************************************************************/
Particle::~Particle()
{
}
/*********************************************************************************************/
/*********************************************************************************************/
Vector3 Particle::getOldPos()
{
	return oldPos;
}
/*********************************************************************************************/
Vector3 Particle::getNewPos()
{
	return newPos;
}
/*********************************************************************************************/
Vector3 Particle::getOldVel()
{
	return oldVel;
}
/*********************************************************************************************/
Vector3 Particle::getNewVel()
{
	return newVel;
}
/*********************************************************************************************/
double  Particle::getMass()
{
	return mass;
}
/*********************************************************************************************/
double  Particle::getParticleRadius()
{
	return particleRadius;
}
/*********************************************************************************************/
Vector3 Particle::getColor()
{
	return color;
}
/*********************************************************************************************/
/*********************************************************************************************/
void Particle::setOldPos(Vector3 oldPos)
{
	this->oldPos = oldPos;
}
/*********************************************************************************************/
void Particle::setNewPos(Vector3 newPos)
{
	this->newPos = newPos;
}
/*********************************************************************************************/
void Particle::setOldVel(Vector3 oldVel)
{
	this->oldVel = oldVel;
}
/*********************************************************************************************/
void Particle::setNewVel(Vector3 newVel)
{
	this->newVel = newVel;
}
/*********************************************************************************************/
void Particle::setMass(double m)
{
	this->mass = m;
}
/*********************************************************************************************/
void Particle::setParticleRadius(double particleRadius)
{
	this->particleRadius = particleRadius;
}
/*********************************************************************************************/
void Particle::setColor(Vector3 color)
{
	this->color = Vector3(color.x(),color.y(),color.z());
}
/*********************************************************************************************/
/*********************************************************************************************/
