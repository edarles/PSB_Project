/* 
 * File:   AnimatedHeightField.cpp
 * Author: mathias
 * 
 * Created on 23 mai 2013, 11:42
 */

#include "AnimatedHeightField.h"

AnimatedHeightField::AnimatedHeightField() {
}

AnimatedHeightField::AnimatedHeightField(const AnimatedHeightField& orig) {
    this->Max = orig.Max;
    this->Min = orig.Min;
    this->center = orig.center;
    this->dx = orig.dx;
    this->dz = orig.dz;
    this->hMax = orig.hMax;
    this->hMin = orig.hMin;
    this->nbPosX = orig.nbPosX;
    this->nbPosY = orig.nbPosY;
    this->nbPosZ = orig.nbPosZ;
    this->pos = orig.pos;
    this->t = orig.t;
}

AnimatedHeightField::~AnimatedHeightField()
{
    
}

void   AnimatedHeightField::calculateHeight(double* m_pos, uint nbPos)
{
    
}

void   AnimatedHeightField::calculateHeight_Normales(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1, uint nbPos)
{
    
}

double AnimatedHeightField::getT() const
{
    return t;
}

void AnimatedHeightField::setT(double t)
{
    this->t = t;
}