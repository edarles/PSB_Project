#include <glDisplay.h>
#include <glWidget.h> 
/**************************************************************************************************************/
/**************************************************************************************************************/
GLDisplay::GLDisplay(GLWidget *widget)
{
	this->glWidget = widget;
	modeParticles = System::POINTS;
	modeFace = GL_FRONT_AND_BACK;
	modeRasterization = GL_LINE;
	drawSurface = false;
	drawNormales_ObjCollision = false;
	drawEmitters = false;
	surfaceSPH = NULL;
	currentObj = NULL;
	colorCollision = Vector3(0,1,0);
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
	this->drawSurface = false;
	this->surfaceSPH = NULL;
	colorCollision = Vector3(0,1,0);
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
bool GLDisplay::getDrawSurface()
{
	return drawSurface;
}
/**************************************************************************************************************/
void GLDisplay::setDrawSurface(bool draw)
{
	drawSurface = draw;
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
	if(glWidget){
		displayParticles();
		displayCollisions();
		displayAnimatedHeightField();
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
	if(system!=NULL){
		system->displayParticles(modeParticles,colorParticles);
		if(glWidget->getSurface()!=NULL)
			glWidget->getSurface()->draw();
		//if(drawSurface)
		//	displaySurface();
	}
}
/**************************************************************************************************************/
/**************************************************************************************************************/
void GLDisplay::displayParticlesByField(uint field){
	System *system = glWidget->getSystem();
	if(system!=NULL){
			system->displayParticlesByField(field);
	}
}
/**************************************************************************************************************/
void GLDisplay::displayCollisions()
{
	System* system = glWidget->getSystem();
	if(system!=NULL)
		system->displayCollisions(modeFace,modeRasterization,colorCollision,colorNormales,drawNormales_ObjCollision);
}
/**************************************************************************************************************/
void GLDisplay::displayEmitters()
{
	System* system = glWidget->getSystem();
	if(system!=NULL)
		system->displayEmitters(colorEmitters);
}
/**************************************************************************************************************/
void GLDisplay::displayAnimatedHeightField()
{
	AnimatedHeightField* AHF = glWidget->getAnimatedHF();
	if(AHF!=NULL){
		glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
		AHF->display(Vector3(1,1,1));
	}
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
void GLDisplay::displaySurface()
{
	System *system = glWidget->getSystem();
        if(system){
		if(typeid(*system)==typeid(SPHSystem) 
		|| typeid(*system)==typeid(PCI_SPHSystem)
		|| typeid(*system)==typeid(MSPHSystem)){
			surfaceSPH = glWidget->getSurface();
			if(surfaceSPH==NULL)
				surfaceSPH = new SurfaceSPH();
			surfaceSPH->extract(system,0.5,20);
			glWidget->setSurface(surfaceSPH);
			surfaceSPH->draw();
		}
		if(typeid(*system)==typeid(SWSPHSystem)){
			SWSPHSystem *S = (SWSPHSystem*) system;
			S->setDisplaySurface(true);
			S->drawSurface();
		}
	}
}
/**************************************************************************************************************/
/**************************************************************************************************************/
