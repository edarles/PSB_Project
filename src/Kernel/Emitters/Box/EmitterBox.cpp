#include <EmitterBox.h>
#include <typeinfo>
#include <Particle.h>
#include <CudaParticle.h>
#include <SphParticle.h>
#include <PciSphParticle.h>

/************************************************************************************************/
/************************************************************************************************/
EmitterBox::EmitterBox():Emitter(),Box()
{
}
/************************************************************************************************/
EmitterBox::EmitterBox(Vector3 center, float sizeX, float sizeY, float sizeZ,Vector3 velocityEmission)
	   :Emitter(Vector3(0,0,0),0,0,1,velocityEmission),Box()
{
	create(center,sizeX,sizeY,sizeZ);
}
/************************************************************************************************/
EmitterBox::EmitterBox(const EmitterBox &C):Emitter(C),Box(C)
{
}
/************************************************************************************************/
EmitterBox::~EmitterBox()
{
}
/************************************************************************************************/
/************************************************************************************************/
vector<Particle*> EmitterBox::emitParticles()
{
	if(data!=NULL){
		vector<Particle*> particles;
		for(float z=(center.z()-sizeZ/2); z<=(center.z()+sizeZ/2); z+=data->getParticleRadius()*2) {
        		for(float y=(center.y()-sizeY/2); y<=(center.y()+sizeY/2); y+=data->getParticleRadius()*2) {
				for(float x=(center.x()-sizeX/2); x<=(center.x()+sizeX/2); x+=data->getParticleRadius()*2) {
					float dx = x + worldPosition.x();
					float dy = y + worldPosition.y();
					float dz = z + worldPosition.z();
					Vector3 pos(dx,dy,dz);
					addParticle(pos,&particles);
				}
            		}
        	}
		currentTime++;
		return particles;
    	}
}
/************************************************************************************************/
/************************************************************************************************/

void EmitterBox::display(Vector3 color)
{
	Box::display(color);
}


