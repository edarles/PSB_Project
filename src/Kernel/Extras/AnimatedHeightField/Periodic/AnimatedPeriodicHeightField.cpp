/* 
 * File:   AnimatedPeriodicHeightField.cpp
 * Author: mathias
 * 
 * Created on 23 mai 2013, 11:47
 */

#include "AnimatedPeriodicHeightField.h"
#include "AnimatedPeriodicHeightField.cuh"

#include <HeightField.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>
#include <GL/gl.h>

using namespace Utils;

AnimatedPeriodicHeightField::AnimatedPeriodicHeightField()
{
    
}

AnimatedPeriodicHeightField::AnimatedPeriodicHeightField(uint nbFunc, double AMin, double AMax, double kMin, double kMax, 
                                                double thetaMin, double thetaMax, double phiMin, double phiMax,
                                                double omegaMin, double omegaMax, double t,
                                                Vector3 Min, Vector3 Max, double d_x, double d_z)
:Periodic_HeightField(nbFunc, AMin, AMax, kMin, kMax, thetaMin, thetaMax, phiMin, phiMax, Min, Max, d_x, d_z)
{
    //Vector3 origin, float length, float width, double dx_, double dz_,
//                        uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
//                         double phiMin, double phiMax, double omegaMin, double omegaMax, double t, float elast
    create(Vector3(0,0,0),3,3,0.05,0.05,nbFunc,AMin,AMax,kMin,kMax,thetaMin,thetaMax,phiMin,phiMax,omegaMin,omegaMax,t,0);
}

AnimatedPeriodicHeightField::AnimatedPeriodicHeightField(const AnimatedPeriodicHeightField& orig)
:Periodic_HeightField(orig)
{
    this->omegaMin = orig.omegaMin;
    this->omegaMax = orig.omegaMax;
    this->t = orig.t;
    this->curTime = orig.curTime;
}

AnimatedPeriodicHeightField::~AnimatedPeriodicHeightField()
{
    
}

void   AnimatedPeriodicHeightField::calculateHeight(double* m_pos, uint nbPos)
{
    AnimatedPeriodicHeightField_calculateHeight_CUDA(m_pos,m_A,m_k,m_theta,m_phi,m_omega,t,nbFunc,nbPos);
}

void   AnimatedPeriodicHeightField::calculateHeight_Normales(double* m_pos, double* nx0, double* nx1, double* nz0, double* nz1, uint nbPos)
{
    AnimatedPeriodicHeightField_calculateHeight_Normales_CUDA(m_pos,nx0,nx1,nz0,nz1,m_A,m_k,m_theta,m_phi,m_omega,t,nbFunc,nbPos);
}

void    AnimatedPeriodicHeightField::create(Vector3 origin, float length, float width, double dx_, double dz_,
                        uint nbFunc, double AMin, double AMax, double kMin, double kMax, double thetaMin, double thetaMax,
                         double phiMin, double phiMax, double omegaMin, double omegaMax, double t, float elast)

{
//    setElast(elast);
	Vector3 min_(origin.x()-length/2,origin.y(),origin.z()-width/2);
	Vector3 max_(origin.x()+length/2,origin.y(),origin.z()+width/2);
    
    this->nbFunc = nbFunc;
    this->AMin = AMin; this->AMax = AMax;
    this->kMin = kMin; this->kMax = kMax;
    this->thetaMin = thetaMin; this->thetaMax = thetaMax;
    this->phiMin = phiMin; this->phiMax = phiMax;
    this->omegaMin = omegaMin; this->omegaMax = omegaMax;
    this->t = t;
    this->curTime = 0.00;
    this->initialize();
    
    Utils::Periodic_HeightField::Min = min_;
	Utils::Periodic_HeightField::Max = max_;
	Utils::Periodic_HeightField::dx = dx_;
	Utils::Periodic_HeightField::dz = dz_;
	Utils::Periodic_HeightField::center = Vector3((Utils::Periodic_HeightField::Max.x()+Utils::Periodic_HeightField::Min.x())/2,(Utils::Periodic_HeightField::Max.y()+Utils::Periodic_HeightField::Min.y())/2,(Utils::Periodic_HeightField::Max.z()+Utils::Periodic_HeightField::Min.z())/2);

    //REPRENDRE CETTE IGNOMINIE
    
    //	Periodic_HeightField::create(nbFunc,AMin,AMax,kMin,kMax,thetaMin,thetaMax,phiMin,phiMax);
    //    	Utils::HeightField::create(min_,max_,dx_,dz_);
    
//    Utils::Periodic_HeightField::generate();
//    	Utils::Periodic_HeightField::display(Vector3(255,255,255));
    
    Periodic_HeightField::create(nbFunc,AMin,AMax,kMin,kMax,thetaMin,thetaMax,phiMin,phiMax);
    this->initialize();
//	Periodic_HeightField::create(min_,max_,dx_,dz_);
//	HeightField::generate();
    curTime += t;

}

void AnimatedPeriodicHeightField::initialize()
{
     
    std::cout<<"initialisation d'un animatedPeriodicHeightField"<<std::endl;
	if(m_A!=NULL) freeArray(m_A);
	if(m_k!=NULL) freeArray(m_k);
	if(m_theta!=NULL) freeArray(m_theta);
	if(m_phi!=NULL) freeArray(m_phi);
	if(m_omega!=NULL) freeArray(m_omega);

        double *A, *k, *theta, *phi, *omega;
	A = new double[nbFunc];
	k = new double[nbFunc];
	theta = new double[nbFunc];
	phi = new double[nbFunc];
	omega = new double[nbFunc];

	srand(time(NULL));
	for(uint i=0;i<nbFunc;i++){
		A[i] = rand()/((double)RAND_MAX )*(AMax-AMin) + AMin;
		k[i] = rand()/((double)RAND_MAX )*(kMax-kMin) + kMin;
		theta[i] = rand()/((double)RAND_MAX )*(thetaMax-thetaMin) + thetaMin;
		phi[i] = rand()/((double)RAND_MAX )*(phiMax-phiMin) + phiMin;
		omega[i] = rand()/((double)RAND_MAX )*(omegaMax-omegaMin) + omegaMin;
		printf("A:%f k:%f theta:%f phi:%f omega:%f\n",A[i],k[i],theta[i],phi[i],omega[i]);
	}
	allocateArray((void**)&m_A, nbFunc*sizeof(double));
	allocateArray((void**)&m_k, nbFunc*sizeof(double));
	allocateArray((void**)&m_theta, nbFunc*sizeof(double));
	allocateArray((void**)&m_phi, nbFunc*sizeof(double));
	allocateArray((void**)&m_omega, nbFunc*sizeof(double));

	copyArrayToDevice(m_A, A, 0, nbFunc*sizeof(double));
	copyArrayToDevice(m_k, k, 0, nbFunc*sizeof(double));
	copyArrayToDevice(m_theta, theta, 0, nbFunc*sizeof(double));
	copyArrayToDevice(m_phi, phi, 0, nbFunc*sizeof(double));
	copyArrayToDevice(m_omega, omega, 0, nbFunc*sizeof(double));

	delete[] A;
	delete[] k;
	delete[] theta;
	delete[] phi;
	delete[] omega;
}

double AnimatedPeriodicHeightField::getCurtime() const {
    return curTime;
}

void AnimatedPeriodicHeightField::setCurtime(double curTime) {
    this->curTime = curTime;
}

void AnimatedPeriodicHeightField::display(Vector3 color)
{
	if(Periodic_HeightField::pos!=NULL){
		for(uint i=0;i<Periodic_HeightField::nbPosX-1;i++){
			for(uint j=0;j<Periodic_HeightField::nbPosZ-1;j++){
				uint index0 = i + j*Periodic_HeightField::nbPosX;
				double xC0 = Periodic_HeightField::pos[index0*3];
				double yC0 = Periodic_HeightField::pos[index0*3+1];
				double zC0 = Periodic_HeightField::pos[index0*3+2];
				uint index1 = i + (j+1)*Periodic_HeightField::nbPosX;
				double xC1 = Periodic_HeightField::pos[index1*3];
				double yC1 = Periodic_HeightField::pos[index1*3+1];
				double zC1 = Periodic_HeightField::pos[index1*3+2];
				uint index2 = (i+1) + (j+1)*Periodic_HeightField::nbPosX;
				double xC2 = Periodic_HeightField::pos[index2*3];
				double yC2 = Periodic_HeightField::pos[index2*3+1];
				double zC2 = Periodic_HeightField::pos[index2*3+2];
				uint index3 = (i+1) + j*Periodic_HeightField::nbPosX;
				double xC3 = Periodic_HeightField::pos[index3*3];
				double yC3 = Periodic_HeightField::pos[index3*3+1];
				double zC3 = Periodic_HeightField::pos[index3*3+2];
				glBegin(GL_TRIANGLES);
				glVertex3f(xC0,yC0,zC0);
				glVertex3f(xC1,yC1,zC1);
				glVertex3f(xC2,yC2,zC2);
				glEnd();
				glBegin(GL_TRIANGLES);
				glVertex3f(xC2,yC2,zC2);
				glVertex3f(xC3,yC3,zC3);
				glVertex3f(xC0,yC0,zC0);
				glEnd();
			}
		}
	}
}

void AnimatedPeriodicHeightField::displayNormale(Vector3 color)
{
    //normale = \nabla f = Vector3(drondf/drondx, drondf/drondy, drondf/drondz)
    Vector3 * N = new Vector3();
    double x, y, z;
    double drondy;
    for(int i=0;i<AMax;i++)
    {
//        drondy += (m_A[i]*m_k[i]*cos(m_theta[i])*sin(m_k[i]*(Periodic_HeightField::pos[i]*cos(m_theta)+Periodic_HeightField::pos[i]->z*sin(m_theta[i]))-m_omega[i]*t+m_phi[i]) );
    }
    drondy *= -1.0;
}