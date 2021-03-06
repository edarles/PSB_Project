/*****************************************************************************************************/
/*****************************************************************************************************/
#ifndef _FORCE_EXT_PERIODIC_
#define _FORCE_EXT_PERIODIC_

#include <ForceExt.h>

/*****************************************************************************************************/
/*****************************************************************************************************/
class ForceExt_Periodic : public ForceExt {

	public:

/*****************************************************************************************************/
/*****************************************************************************************************/
		ForceExt_Periodic();
		ForceExt_Periodic(float amplitude, float longueurOnde, float deviation, float frequency, float dephasage);
		ForceExt_Periodic(const ForceExt_Periodic& F);
		~ForceExt_Periodic();

		void draw();

/*****************************************************************************************************/
/*****************************************************************************************************/
		float getAmplitude();
		float getLongueurOnde();
		float getNbreOnde();
		float getDeviation();
		float getPulsation();
		float getFrequency();
		float getDephasage();
		float getTime();

/*****************************************************************************************************/
/*****************************************************************************************************/
		void setAmplitude(float);
		void setLongueurOnde(float);
		void setNbreOnde(float);
		void setDeviation(float);
		void setPulsation(float);
		void setFrequency(float);
		void setDephasage(float);
		void setTime(float);

/*****************************************************************************************************/
/*****************************************************************************************************/
	private:

		float A;
		float lambda;
		float k;
		float theta;
		float w;
		float f;
		float phi;
		float time;
};
/*****************************************************************************************************/

#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
