#include <Sphere.h>

/***********************************************************************************/
/***********************************************************************************/
namespace Utils
{
/***********************************************************************************/
/***********************************************************************************/
Sphere::Sphere():ObjectGeo()
{
	center = Vector3(0,0,0);
	radius = 1.0;
}
/***********************************************************************************/
Sphere::Sphere(Vector3 center, float radius):ObjectGeo()
{
	this->center = center;
	this->radius = radius;
}
/***********************************************************************************/
Sphere::Sphere(const Sphere &S):ObjectGeo(S)
{
	this->center = S.center;
	this->radius = S.radius;
}
/***********************************************************************************/
Sphere::~Sphere()
{
}
/***********************************************************************************/
/***********************************************************************************/
Vector3 Sphere::getCenter()
{
	return center;
}
/***********************************************************************************/
float   Sphere::getRadius()
{
	return radius;
}
/***********************************************************************************/
/***********************************************************************************/
void	Sphere::setCenter(Vector3 center)
{
	this->center = center;
}
/***********************************************************************************/
void	Sphere::setRadius(float radius)
{
	this->radius = radius;
}
/***********************************************************************************/
/***********************************************************************************/
void	Sphere::display(Vector3 color)
{
	glColor3f(color.x(),color.y(),color.z());
	GLUquadric *Boule = gluNewQuadric();
	glTranslated(center.x(),center.y(),center.z());
	gluSphere(Boule, radius, 20, 20);
	glTranslated(-center.x(),-center.y(),-center.z());
	gluDeleteQuadric(Boule);
	glColor3f(1.0,1.0,1.0);
}
/***********************************************************************************/
void 	Sphere::displayNormale(Vector3 color)
{
}
/***********************************************************************************/
/***********************************************************************************/
}
/***********************************************************************************/
/***********************************************************************************/
