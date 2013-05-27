/* 
 * File:   AnimatedPeriodicHeightField.h
 * Author: mathias
 *
 * Created on 23 mai 2013, 11:47
 */

#ifndef ANIMATEDPERIODICHEIGHTFIELD_H
#define	ANIMATEDPERIODICHEIGHTFIELD_H

#include <AnimatedHeightField.h>
#include <PeriodicHeightField.h>
#include <ObjectCollision.h>

class AnimatedPeriodicHeightField : public AnimatedHeightField, public Utils::Periodic_HeightField
{
public:
    AnimatedPeriodicHeightField();
    AnimatedPeriodicHeightField(uint nbFunc, double AMin, double AMax, double kMin, double kMax, 
                                                double thetaMin, double thetaMax, double phiMin, double phiMax,
                                                double omegaMin, double omegaMax, double t,
                                                Vector3 Min, Vector3 Max, double d_x, double d_z);
    AnimatedPeriodicHeightField(const AnimatedPeriodicHeightField& orig);
    virtual ~AnimatedPeriodicHeightField();
    
    virtual void   calculateHeight(double* m_pos, uint nbPos);
    virtual void   calculateHeight_Normales(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1, uint nbPos);
    void    create(Vector3 origin, float length, float width, double dx_, double dz_,
                        uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
                         double phiMin, double phiMax, double omegaMin, double omegaMax, double t, float elast);
    
    void initialize();
    
    double getCurtime() const;


    void setCurtime(double curTime);
    
    void display(Vector3 color);
    
    void displayNormale(Vector3 color);
    
private:
    //gpu store
    double *m_omega;
    
    double curTime; //to get the current step (sum of all timesteps done)
    
    double omegaMin, omegaMax;
};

#endif	/* ANIMATEDPERIODICHEIGHTFIELD_H */

