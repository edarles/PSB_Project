/* 
 * File:   AnimatedHeightField.h
 * Author: mathias
 *
 * Created on 23 mai 2013, 11:42
 */

#ifndef ANIMATEDHEIGHTFIELD_H
#define	ANIMATEDHEIGHTFIELD_H

#include <HeightField.h>

class AnimatedHeightField : public Utils::HeightField {
public:
    AnimatedHeightField();
    AnimatedHeightField(const AnimatedHeightField& orig);
    virtual ~AnimatedHeightField();
    
    virtual void   calculateHeight(double* m_pos, uint nbPos) = 0;
    virtual void   calculateHeight_Normales(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1, uint nbPos) = 0;
    
    double getT() const;
    void setT(double t);
    
    double getCurTime() const;
    void setCurTime(double cur_time);
    
    void update();
    
protected:
    double t;
    double curTime;
};

#endif	/* ANIMATEDHEIGHTFIELD_H */

