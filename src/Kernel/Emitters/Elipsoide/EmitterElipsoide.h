#ifndef _EMITTER_ELIPSOIDE_
#define _EMITTER_ELIPSOIDE_

#include <Emitter.h>

class EmitterElipsoide : public Emitter 
{
	public:
		EmitterElipsoide();
		EmitterElipsoide(Vector3 center, float sizeX, float sizeZ, unsigned int minEmission, 
				  unsigned int maxEmission, unsigned int durationTime, Vector3 velocityEmission);

		EmitterElipsoide(const EmitterElipsoide&);
		~EmitterElipsoide();

		Vector3 getCenter();
		float   getSizeX();
		float   getSizeZ();

		void    setCenter(Vector3 center);
		void    setSizeX(float sizeX);
		void    setSizeZ(float sizeZ);

		vector<Particle*> emitParticles();
		void		  display(Vector3 color);

	private:
		Vector3 center;
		float   sizeX, sizeZ;
	
};
#endif
