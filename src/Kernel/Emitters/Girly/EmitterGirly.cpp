#include <EmitterGirly.h>
#include <stdio.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>
/************************************************************************************************/
/************************************************************************************************/
EmitterGirly::EmitterGirly():Emitter()
{
}
/************************************************************************************************/
EmitterGirly::EmitterGirly(Vector3 velocityEmission)
	    :Emitter(Vector3(0,0,0),1,1,1,velocityEmission)
{
}
/************************************************************************************************/
EmitterGirly::EmitterGirly(const EmitterGirly& M):Emitter(M)
{
}
/************************************************************************************************/
EmitterGirly::~EmitterGirly()
{
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterGirly::emitParticles()
{
	if(data!=NULL){
	vector<Particle*> particles;
	if(currentTime<durationTime){
		double res;
		double eps = 0.001;
		double step = 0.02;
		for(double x = -3.0; x<=3.0; x+=step){
			for(double y = -3.0; y<=3.0; y+=step){
				for(double z=-3.0; z<=3.0; z+=step){
					res = powf((x*x + (9/4)*y*y + z*z -1),3) - x*x*z*z*z - (9/80)*y*y*z*z*z;
					if(res>=-eps & res<=eps){
						Vector3 pos = Vector3(x,y,z);
						addParticle(pos,&particles);
					}
				}
			}
		}
		currentTime++;
	}
	return particles;
	}
}
/************************************************************************************************/
/************************************************************************************************/
void EmitterGirly::display(Vector3 color)
{
	//Sphere::display(color);
}
/************************************************************************************************/
/************************************************************************************************/
