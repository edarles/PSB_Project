/*****************************************************************************************************/
/*****************************************************************************************************/
#include <ForceExt_Periodic.h>

/*****************************************************************************************************/
/*****************************************************************************************************/
ForceExt_Periodic::ForceExt_Periodic():ForceExt()
{
	A = 0;
	lambda = 0;
	k = 0;
	theta = 0;
	w = 0;
	f = 0;
	phi = 0;
	time = 0;
}
/*****************************************************************************************************/
ForceExt_Periodic::ForceExt_Periodic(float A, float lambda, float theta, float f, float phi):ForceExt()
{
	this->A = A;
	this->lambda = lambda;
	this->theta = theta;
	this->f = f;
	this->phi = phi;
	this->k = 2*M_PI/lambda;
	this->w = 2*M_PI*f;
	this->time = 0;
}
/*****************************************************************************************************/
ForceExt_Periodic::ForceExt_Periodic(const ForceExt_Periodic& F):ForceExt()
{
	this->A = F.A;
	this->lambda = F.lambda;
	this->theta = F.theta;
	this->f = F.f;
	this->phi = F.phi;
	this->k = F.k;
	this->w = F.w;
	this->time = F.time;
}
/*****************************************************************************************************/
ForceExt_Periodic::~ForceExt_Periodic()
{
}

/*****************************************************************************************************/
void ForceExt_Periodic::draw()
{
}
/*****************************************************************************************************/
/*****************************************************************************************************/
float ForceExt_Periodic::getAmplitude()
{
	return A;
}
/*****************************************************************************************************/
float ForceExt_Periodic::getLongueurOnde()
{
	return lambda;
}
/*****************************************************************************************************/
float ForceExt_Periodic::getNbreOnde()
{
	return k;
}
/*****************************************************************************************************/
float ForceExt_Periodic::getDeviation()
{
	return theta;
}
/*****************************************************************************************************/
float ForceExt_Periodic::getPulsation()
{
	return w;
}
/*****************************************************************************************************/
float ForceExt_Periodic::getFrequency()
{
	return f;
}
/*****************************************************************************************************/
float ForceExt_Periodic::getDephasage()
{
	return phi;
}
/*****************************************************************************************************/
float ForceExt_Periodic::getTime()
{
	return time;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void ForceExt_Periodic::setAmplitude(float A)
{
	this->A = A;
}
/*****************************************************************************************************/
void ForceExt_Periodic::setLongueurOnde(float lambda)
{
	this->lambda = lambda;
}
/*****************************************************************************************************/
void ForceExt_Periodic::setNbreOnde(float k)
{
	this->k =k;
}
/*****************************************************************************************************/
void ForceExt_Periodic::setDeviation(float theta)
{
	this->theta = theta;
}
/*****************************************************************************************************/
void ForceExt_Periodic::setPulsation(float w)
{
	this->w = w;
}
/*****************************************************************************************************/
void ForceExt_Periodic::setFrequency(float f)
{
	this->f = f;
}
/*****************************************************************************************************/
void ForceExt_Periodic::setDephasage(float phi)
{
	this->phi = phi;
}
/*****************************************************************************************************/
void ForceExt_Periodic::setTime(float time)
{
	this->time = time;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
