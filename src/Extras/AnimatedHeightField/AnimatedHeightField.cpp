/* 
 * File:   AnimatedHeightField.cpp
 * Author: mathias
 * 
 * Created on 23 mai 2013, 11:42
 */
/****************************************************************************************/
/****************************************************************************************/
#include <AnimatedHeightField.h>
#include <common.cuh> 
#include <stdio.h>
#include <sstream>
#include <vector>
/****************************************************************************************/
/****************************************************************************************/
AnimatedHeightField::AnimatedHeightField()
{
	t = 0;
	step = 0.1;
}
/****************************************************************************************/
AnimatedHeightField::AnimatedHeightField(double step)
{
	t = 0;
	this->step = step;
}
/****************************************************************************************/
AnimatedHeightField::AnimatedHeightField(const AnimatedHeightField& H)
{
        this->t = H.t;
	this->step = H.step;
}
/****************************************************************************************/
AnimatedHeightField::~AnimatedHeightField()
{    
}
/****************************************************************************************/
/****************************************************************************************/
double AnimatedHeightField::getT() const
{
	return t;
}
/****************************************************************************************/
double AnimatedHeightField::getStep() const
{
	return step;
}
/****************************************************************************************/
/****************************************************************************************/
void   AnimatedHeightField::setStep(double step)
{
	this->step = step;
}
