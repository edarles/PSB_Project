#include "ObjectCollision.h"
/*************************************************************************************/
/*************************************************************************************/
ObjectCollision::ObjectCollision()
{
	elast = 0;
	friction = 0;
	is_container = false;
}
/*************************************************************************************/
ObjectCollision::ObjectCollision(float elast, float friction, bool is_container)
{
	this->elast = elast;
	this->friction = friction;
	this->is_container = is_container;
}
/*************************************************************************************/
ObjectCollision::ObjectCollision(const ObjectCollision& O)
{
	this->elast = O.elast;
	this->friction = friction;
	this->is_container = O.is_container;
}
/*************************************************************************************/
ObjectCollision::~ObjectCollision()
{
}
/*************************************************************************************/
/*************************************************************************************/
float ObjectCollision::getElast()
{
	return elast;
}
/*************************************************************************************/
float ObjectCollision::getFriction()
{
	return friction;
}
/*************************************************************************************/
bool ObjectCollision::getIsContainer()
{
	return is_container;
}
/*************************************************************************************/
/*************************************************************************************/
void  ObjectCollision::setElast(float elast)
{
	this->elast = elast;
}
/*************************************************************************************/
void  ObjectCollision::setFriction(float friction)
{
	this->friction = friction;
}
/*************************************************************************************/
void  ObjectCollision::setIsContainer(bool is_container)
{
	this->is_container = is_container;
}
/*************************************************************************************/
/*************************************************************************************/
