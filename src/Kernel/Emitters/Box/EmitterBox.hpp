#include <EmitterBox.h>

/************************************************************************************************/
/************************************************************************************************/
template class <T>
EmitterBox<T>::EmitterBox():Emitter(),Box()
{
	data = new T();
}
/************************************************************************************************/
template class <T>
EmitterBox<T>::EmitterBox(Vector3 center, float sizeX, float sizeY, float sizeZ,Vector3 velocityEmission)
	   :Emitter(Vector3(0,0,0),0,0,1,velocityEmission),Box()
{
	create(center,sizeX,sizeY,sizeZ);
	data = new T();
}
/************************************************************************************************/
template class<T>
EmitterBox<T>::EmitterBox(const EmitterBox<T> &C):Emitter(C),Box(C)
{
	if(data==NULL)
		data = new T();
	data = C.data;
}
/************************************************************************************************/
template class <T>
EmitterBox<T>::~EmitterBox()
{
	delete(data);
}
/************************************************************************************************/
/************************************************************************************************/
template class <T>
T* EmitterBox<T>::getData()
{
	return data;
]
/************************************************************************************************/
template class <T>
void EmitterBox<T>::setData(T* data)
{
	if(this->data==NULL)
		this->data = new T();
	this->data = data;
}
/************************************************************************************************/
/************************************************************************************************/
template class <T>
vector<Particle*> EmitterBox<T>::emitParticles()
{
	if(data!=NULL){
		vector<Particle*> particles;
		for(float z=(center.z()-sizeZ/2); z<=(center.z()+sizeZ/2); z+=data->getParticleRadius()*2) {
        		for(float y=(center.y()-sizeY/2); y<=(center.y()+sizeY/2); y+=data->getParticleRadius()*2) {
				for(float x=(center.x()-sizeX/2); x<=(center.x()+sizeX/2); x+=data->getParticleRadius()*2) {
				float dx = x + worldPosition.x();
				float dy = y + worldPosition.y();
				float dz = z + worldPosition.z();
				Particle *p = new Particle(Vector3(dx,dy,dz),velocityEmission, data->getParticleMass());
				particles.push_back(p);
            		}
        	}
    	}
	currentTime++;
	return particles;
}
/************************************************************************************************/
/************************************************************************************************/
template class <T>
void EmitterBox<T>::display(Vector3 color)
{
	Box::display(color);
}


