#include <EmitterSphere.h>
#include <stdio.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>
/************************************************************************************************/
/************************************************************************************************/
EmitterSphere::EmitterSphere():Emitter(),Sphere()
{
}
/************************************************************************************************/
EmitterSphere::EmitterSphere(const Sphere& M, Vector3 velocityEmission)
	    :Emitter(Vector3(0,0,0),1,1,1,velocityEmission),Sphere(M)
{
}
/************************************************************************************************/
EmitterSphere::EmitterSphere(Vector3 center, float baseRadius, Vector3 velocityEmission)
	      :Emitter(Vector3(0,0,0),1,1,1,velocityEmission),Sphere(center,baseRadius)
{
}
/************************************************************************************************/
EmitterSphere::EmitterSphere(const EmitterSphere& M):Emitter(M),Sphere(M)
{
}
/************************************************************************************************/
EmitterSphere::~EmitterSphere()
{
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterSphere::emitParticles()
{
	if(data!=NULL){
	vector<Particle*> particles;
	if(currentTime<durationTime){
		double y = center.y();
		while(y<center.y()+radius){
			double r = 0;
			while(r<=sqrt(radius*radius - y*y)){
				double theta = 0;
				double thetaAp = 1;
				double dTheta = acos(r/sqrt(r*r+2*data->getParticleRadius()*data->getParticleRadius()));
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

		y = center.y()-data->getParticleRadius();
		while(y>=center.y()-radius){
			double r = 0;
			while(r<=sqrt(radius*radius - y*y)){
				double theta = 0;
				double thetaAp = 1;
				double dTheta = acos(r/sqrt(r*r+2*data->getParticleRadius()*data->getParticleRadius()));
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
			y -= data->getParticleRadius();
		}
		currentTime++;
	}
	return particles;
	}
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterSphere::display(Vector3 color)
{
	Sphere::display(color);
}
/************************************************************************************************/
/************************************************************************************************/
