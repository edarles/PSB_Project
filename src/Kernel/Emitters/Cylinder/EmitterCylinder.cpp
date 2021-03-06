#include <EmitterCylinder.h>
#include <stdio.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>
/************************************************************************************************/
/************************************************************************************************/
EmitterCylinder::EmitterCylinder():Emitter(),Cylinder()
{
}
/************************************************************************************************/
EmitterCylinder::EmitterCylinder(const Cylinder& M, Vector3 velocityEmission)
	    :Emitter(Vector3(0,0,0),1,1,1,velocityEmission),Cylinder(M)
{
}
/************************************************************************************************/
EmitterCylinder::EmitterCylinder(Vector3 center, float baseRadius, float length, Vector3 direction, Vector3 velocityEmission)
		:Emitter(Vector3(0,0,0),1,1,1,velocityEmission),Cylinder(center,baseRadius,length,direction)
{
}
/************************************************************************************************/
EmitterCylinder::EmitterCylinder(const EmitterCylinder& M):Emitter(M),Cylinder(M)
{
}
/************************************************************************************************/
EmitterCylinder::~EmitterCylinder()
{
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterCylinder::emitParticles()
{
	if(data!=NULL){
	vector<Particle*> particles;
	if(currentTime<durationTime){
		Vector3 P1 = Vector3(center.x() + direction.x()*(length/2),
				     center.y() + direction.y()*(length/2),
				     center.z() + direction.z()*(length/2));
		Vector3 P2 = Vector3(center.x() - direction.x()*(length/2),
				     center.y() - direction.y()*(length/2),
				     center.z() - direction.z()*(length/2));
		double y = P2.y();
		while(y<P1.y()){
			double r = 0;
			while(r<=baseRadius){
			double theta = 0;
			double thetaAp = 1;
			double dTheta = acos(r/sqrt(r*r+data->getParticleRadius()*data->getParticleRadius()));
			while (theta<=2*M_PI && thetaAp>0){	
				double x = center.x() + r*cos(theta);
				double z = center.z() + r*sin(theta);
				Vector3 pos = Vector3(x,y,z);
				theta += dTheta;
				thetaAp = theta;
				addParticle(pos,&particles);
			}
			r += data->getParticleRadius();
			}
			y += data->getParticleRadius();
		}
		currentTime++;
	}
	return particles;
	}
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterCylinder::display(Vector3 color)
{
	Cylinder::display(color);
}
/************************************************************************************************/
/************************************************************************************************/
