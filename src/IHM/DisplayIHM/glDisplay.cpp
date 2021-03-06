#include <glDisplay.h>
#include <glWidget.h> 

#include "AnimatedHeightField.h" 
/**************************************************************************************************************/
/**************************************************************************************************************/
GLDisplay::GLDisplay(GLWidget *widget)
{
	this->glWidget = widget;
	modeParticles = System::POINTS;
	modeFace = GL_FRONT_AND_BACK;
	modeRasterization = GL_LINE;
	drawNormales_ObjCollision = false;
	drawEmitters = false;
	currentObj = NULL;
	colorCollision = Vector3(1,1,1);
	colorParticles = Vector3(1,1,1);
	colorNormales  = Vector3(1,0,0);
	colorEmitters  = Vector3(0,0,1);
}
/**************************************************************************************************************/
GLDisplay::GLDisplay(GLWidget *widget, System::ParticleDisplay modeParticles, GLenum modeFace, GLenum modeRasterization, bool drawNormales_ObjCollision)
{
	this->glWidget = widget;
	this->modeParticles = modeParticles;
	this->modeFace = modeFace;
	this->modeRasterization = modeRasterization;
	this->drawNormales_ObjCollision = drawNormales_ObjCollision;
	this->drawEmitters = false;
	colorCollision = Vector3(1,1,1);
	colorParticles = Vector3(1,1,1);
	colorNormales  = Vector3(1,0,0);
	colorEmitters  = Vector3(0,0,1);
	currentObj = NULL;
}
/**************************************************************************************************************/
GLDisplay::~GLDisplay()
{
}
/**************************************************************************************************************/
/**************************************************************************************************************/
GLWidget* GLDisplay::getGLWidget()
{
	return glWidget;
}
/**************************************************************************************************************/
System::ParticleDisplay GLDisplay::getModeParticles()
{
	return modeParticles;
}
/**************************************************************************************************************/
GLenum GLDisplay::getModeFace()
{
	return modeFace;
}
/**************************************************************************************************************/
GLenum GLDisplay::getModeRasterization()
{
	return modeRasterization;
}
/**************************************************************************************************************/
bool GLDisplay::getDrawNormales_ObjCollision()
{
	return drawNormales_ObjCollision;
}
/**************************************************************************************************************/
bool GLDisplay::getDrawEmitters()
{
	return drawEmitters;
 }
/**************************************************************************************************************/
/**************************************************************************************************************/
void GLDisplay::setGLWidget(GLWidget* glWidget)
{
	this->glWidget = glWidget;
}
/**************************************************************************************************************/
void GLDisplay::setModeParticles(System::ParticleDisplay modeParticles)
{
	this->modeParticles = modeParticles;
}
/**************************************************************************************************************/
void GLDisplay::setModeFace(GLenum modeFace)
{
	this->modeFace = modeFace;
}
/**************************************************************************************************************/
void GLDisplay::setModeRasterization(GLenum modeRasterization)
{
	this->modeRasterization = modeRasterization;
}
/**************************************************************************************************************/
void GLDisplay::setDrawNormales_ObjCollision(bool isDraw)
{
	drawNormales_ObjCollision = isDraw;
}
/**************************************************************************************************************/
void GLDisplay::setDrawEmitters(bool isDraw)
{
	drawEmitters = isDraw;
}
/**************************************************************************************************************/
void GLDisplay::setObjectIsCreate()
{
	currentObj = NULL;
}
/**************************************************************************************************************/
/**************************************************************************************************************/
void GLDisplay::display()
{
	glClearColor(0,0,0,1.0 );
	if(glWidget){
		displayParticles();
		displayCollisions();
		if(currentObj)
			displayObjectCollision(currentObj);
		if(drawEmitters)
			displayEmitters();
	}
}
/**************************************************************************************************************/
/**************************************************************************************************************/
void GLDisplay::displayParticles()
{
	System *system = glWidget->getSystem();
	if(system){
		system->displayParticles(modeParticles,colorParticles);
		glWidget->displaySurface();
	}
}
/**************************************************************************************************************/
void GLDisplay::displayCollisions()
{
	System* system = glWidget->getSystem();
	if(system)
		system->displayCollisions(modeFace,modeRasterization,colorCollision,colorNormales,drawNormales_ObjCollision);
}
/**************************************************************************************************************/
void GLDisplay::displayEmitters()
{
	System* system = glWidget->getSystem();
	if(system)
		system->displayEmitters(colorEmitters);
}
/**************************************************************************************************************/
/**************************************************************************************************************/
void GLDisplay::displayAnimatedHeightFields()
{
//    System *system = glWidget->getSystem();
//    if(system)
//        system->displayAnimatedHeightFields(Vector3(255,255,255));
    
}
/**************************************************************************************************************/
/**************************************************************************************************************/
void GLDisplay::displayObjectCollision(ObjectCollision *O)
{
	currentObj = O;
	O->display(modeParticles,modeRasterization, colorCollision);
	if(drawNormales_ObjCollision)
		O->displayNormales(colorNormales);
}
/**************************************************************************************************************/
/**************************************************************************************************************/
void GLDisplay::displayEmitter(Emitter* E)
{
	glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	E->display(colorEmitters);
}
/**************************************************************************************************************/
/**************************************************************************************************************/
