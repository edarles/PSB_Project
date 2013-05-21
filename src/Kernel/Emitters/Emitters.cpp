#include <Emitters.h>
/**********************************************************************************/
/**********************************************************************************/
Emitters::Emitters()
{
	emitters.clear();
}
/**********************************************************************************/
Emitters::~Emitters()
{
	for(unsigned int i=0;i<emitters.size();i++)
		delete(emitters[i]);
	emitters.clear();
}
/**********************************************************************************/
/**********************************************************************************/
vector<Emitter*> Emitters::getEmitters()
{
	return emitters;
}
/**********************************************************************************/
Emitter*	Emitters::getEmitter(unsigned int i)
{
	assert(i<emitters.size());
	return emitters[i];
}
/**********************************************************************************/
/**********************************************************************************/
void	Emitters::setEmitters(vector<Emitter*> emitters)
{
	this->emitters.clear();
	for(unsigned int i=0;i<emitters.size();i++)
		this->emitters.push_back(emitters[i]);
}
/**********************************************************************************/
void	Emitters::setEmitter(unsigned int i, Emitter* E)
{
	assert(i<emitters.size());
	emitters[i] = E;
}
/**********************************************************************************/
void	Emitters::addEmitter(Emitter* E)
{
	emitters.push_back(E);
}
/**********************************************************************************/
/**********************************************************************************/
vector<Particle*> Emitters::emitParticles()
{
	vector<Particle*> result;
	for(unsigned int i=0;i<emitters.size();i++){
		if(emitters[i]->getDurationTime()>emitters[i]->getCurrentTime()){
			vector<Particle*> resultI = emitters[i]->emitParticles();
			result.insert(result.end(), resultI.begin(), resultI.end());
		}
	}
	return result;
}
/**********************************************************************************/
/**********************************************************************************/
void	Emitters::display(Vector3 color)
{
	for(unsigned int i=0;i<emitters.size();i++)
		emitters[i]->display(color);
}
/**********************************************************************************/
/**********************************************************************************/
