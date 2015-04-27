#include <EmitterElipsoide.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/************************************************************************************************/
/************************************************************************************************/
EmitterElipsoide::EmitterElipsoide():Emitter()
{
	center = Vector3(0,0,0);
	radius = 0.2;
	direction = Vector3(0,1,0);
}
/************************************************************************************************/
EmitterElipsoide::EmitterElipsoide(Vector3 center, float radius,  double dx, double dy, double dz, unsigned int durationTime, 
				   Vector3 velocityEmission)
		 :Emitter(Vector3(0,0,0),1,1,durationTime,velocityEmission)
{
	this->center = center;
	this->radius = radius;
	this->direction = Vector3(dx,dy,dz);
}
/************************************************************************************************/
EmitterElipsoide::EmitterElipsoide(const EmitterElipsoide& E): Emitter(E)
{
	this->center = E.center;
	this->radius = E.radius;
	this->direction = E.direction;
}
/************************************************************************************************/
EmitterElipsoide::~EmitterElipsoide()
{
}
/************************************************************************************************/
/************************************************************************************************/
Vector3 EmitterElipsoide::getCenter()
{
	return center;
}
/************************************************************************************************/
float   EmitterElipsoide::getRadius()
{
	return radius;
}
/************************************************************************************************/
Vector3   EmitterElipsoide::getDirection()
{
	return direction;
}
/************************************************************************************************/
/************************************************************************************************/
void	EmitterElipsoide::setCenter(Vector3 center)
{
	this->center = center;
}
/************************************************************************************************/
void	EmitterElipsoide::setRadius(float radius)
{
	this->radius = radius;
}
/************************************************************************************************/
void	EmitterElipsoide::setDirection(Vector3 direction)
{
	this->direction = direction;
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterElipsoide::emitParticles()
{
	vector<Particle*> particles;
	double r = 0;
	while(r<=radius){
		double theta = 0;
		double thetaAp = 1;
		double dTheta = acos(r/sqrt(r*r+data->getParticleRadius()*data->getParticleRadius()));
		if(r!=0){
		while (theta<=2*M_PI && thetaAp>0){	
			double x = center.x() + r*cos(theta)*(1-fabs(direction.x()));
			double y = center.y() + r*cos(theta)*(1-fabs(direction.y()));
			double z = center.z() + r*sin(theta)*(1-fabs(direction.z()));
			Vector3 pos = Vector3(x,y,z);
			theta += dTheta;
			thetaAp = theta;
			addParticle(pos,&particles);
		}
		}
		r += 2.5*data->getParticleRadius();
		//r += 2.5*data->getInteractionRadius()
	}
	currentTime++;
	return particles;
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterElipsoide::emitParticles2D()
{
	vector<Particle*> particles;
	double r = 0;
	while(r<=radius){
		double theta = 0;
		double thetaAp = 1;
		double dTheta = acos(r/sqrt(r*r+data->getParticleRadius()*data->getParticleRadius()));
		if(r!=0){
		while (theta<=2*M_PI && thetaAp>0){	
			double x = center.x() + r*cos(theta)*(1-fabs(direction.x()));
			double y = center.y() + r*cos(theta)*(1-fabs(direction.y()));
			double z = center.z();
			Vector3 pos = Vector3(x,y,z);
			theta += dTheta;
			thetaAp = theta;
			addParticle(pos,&particles);
		}
		}
		r += 2.5*data->getParticleRadius();
	}
	currentTime++;
	return particles;
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterElipsoide::emitParticles2D_z()
{
	vector<Particle*> particles;
	double r = 0;
	while(r<=radius){
		double theta = 0;
		double thetaAp = 1;
		double dTheta = acos(r/sqrt(r*r+data->getParticleRadius()*data->getParticleRadius()));
		if(r!=0){
		while (theta<=2*M_PI && thetaAp>0){	
			double x = center.x() + r*cos(theta)*(1-fabs(direction.x()));
			double y = center.y();
			double z = center.z() + r*cos(theta)*(1-fabs(direction.z()));
			Vector3 pos = Vector3(x,y,z);
			theta += dTheta;
			thetaAp = theta;
			addParticle(pos,&particles);
		}
		}
		r += 2.5*data->getParticleRadius();
	}
	currentTime++;
	return particles;
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterElipsoide::display(Vector3 color)
{
	glColor3f(color.x(),color.y(),color.z());
	glColor3f(1,1,1);
	if(direction.y()==1 || direction.y()==-1)glRotatef(90,1,0,0);
	if(direction.z()==1 || direction.z()==-1)glRotatef(90,0,1,0);
	if(direction.x()==1 || direction.x()==-1)glRotatef(90,0,0,1);
	//glScalef(sizeX,1,sizeZ);
	glTranslatef(center.x(),center.y(),center.z());
	GLUquadric *qobj = gluNewQuadric();
	gluDisk(qobj,radius,radius,20,20);
	//glScalef(-sizeX,-1,-sizeZ);
	glTranslatef(-center.x(),-center.y(),-center.z());
        if(direction.y()==1 || direction.y()==-1)glRotatef(90,-1,0,0);
	if(direction.z()==1 || direction.z()==-1)glRotatef(90,0,-1,0);
	if(direction.x()==1 || direction.x()==-1)glRotatef(90,0,0,-1);
 	glColor3f(1,1,1);
}

