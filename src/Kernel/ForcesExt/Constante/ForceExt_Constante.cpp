#include <ForceExt_Constante.h>
#include <ForceExt_Constante.cuh>
/*****************************************************************************************************/
/*****************************************************************************************************/
ForceExt_Constante::ForceExt_Constante():ForceExt()
{
	direction = Vector3(0,0,0);
	amplitude = 0;
}
/*****************************************************************************************************/
ForceExt_Constante::ForceExt_Constante(Vector3 direction, float amplitude):ForceExt()
{
	this->direction = direction;
	this->amplitude = amplitude;
}
/*****************************************************************************************************/
ForceExt_Constante::ForceExt_Constante(const ForceExt_Constante &F):ForceExt(F)
{
	this->direction = direction;
	this->amplitude = amplitude;
}
/*****************************************************************************************************/
ForceExt_Constante::~ForceExt_Constante()
{
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void ForceExt_Constante::evaluate(double* pos, double* acummForce, double* mass, uint nbBodies)
{
	evaluate_force_constante_CUDA (acummForce, mass, direction, amplitude, nbBodies);
}	
/*****************************************************************************************************/
/*****************************************************************************************************/
Vector3 ForceExt_Constante::getDirection()
{
	return direction;
}
/*****************************************************************************************************/
float   ForceExt_Constante::getAmplitude()
{
	return amplitude;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
void ForceExt_Constante::setDirection(Vector3 direction)
{
	this->direction = direction;
}
/*****************************************************************************************************/
void ForceExt_Constante::setAmplitude(float amplitude)
{
	this->amplitude = amplitude;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
