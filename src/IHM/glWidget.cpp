#include <GL/glew.h>
#include "glWidget.h"
#include <iostream>
#include <unistd.h>
//mencoder mf://*.jpg -mf w=640:h=320:fps=25:type=jpeg -ovc lavc -o video.avi
//xwininfo
//recordmydesktop --windowid 0x264bc09

//******************************************************************************
//******************************************************************************
System* GLWidget::getSystem(){
 return system;
}
//******************************************************************************
void GLWidget::setSystem(System *system){
 this->system = system;
}
//******************************************************************************
//******************************************************************************
AnimatedHeightField* GLWidget::getAnimatedHF()
{
	return AHF;
}
//******************************************************************************
void  GLWidget::setAnimatedHF(AnimatedHeightField* HF)
{
	this->AHF = HF;
}
//******************************************************************************
//******************************************************************************
SurfaceSPH* GLWidget::getSurface()
{
	return surface;
}
//******************************************************************************
//******************************************************************************
GLDisplay* GLWidget::getDisplay()
{
	return display;
}
//******************************************************************************
void	GLWidget::setDisplay(GLDisplay *display)
{
	this->display = display;
}
//******************************************************************************
//******************************************************************************
QSize GLWidget::minimumSizeHint() const {
	return QSize(640, 480);
}
//******************************************************************************
QSize GLWidget::sizeHint() const {
	return QSize(1024, 768);
}
//******************************************************************************
//******************************************************************************
void GLWidget::init() 
{
	captureImage = false;
	frame_courante = 0;
	system = NULL;
        AHF = NULL;
	surface = NULL;
	display = new GLDisplay(this);
       	setAxisIsDrawn(false);
//	setFPSIsDisplayed(true);
	setTextIsEnabled (true);
	//setGridIsDrawn();
	setShortcut(ANIMATION, Qt::CTRL + Qt::Key_A);
	camera()->setType(Camera::PERSPECTIVE);
	camera()->setSceneRadius(1.0);
	camera()->showEntireScene();

	initRender();
}

//******************************************************************************
//******************************************************************************
void GLWidget::initRender() 
{
	
  Vector3 light[2], light_to[2];				// Light stuff
  float     light_fov, cam_fov;

	glewInit();

	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
	srand ( time ( 0x0 ) );
	glClearColor( 0,0,0, 1.0 );
	glShadeModel( GL_SMOOTH );
	glEnable ( GL_COLOR_MATERIAL );
	glEnable (GL_DEPTH_TEST);
	glDepthMask ( 1 );
	
        light[0].setXYZ(39,-60,43);
	light_to[0].setXYZ(0,0,0);     
	light[1].setXYZ(15,-5,145); 
	light_to[1].setXYZ(0,0,0);            
	light_fov = 45;
}

//******************************************************************************
//******************************************************************************
void GLWidget::initSystem() 
{
  //assert(system!=NULL);
  //assert(system->getSimulationData()!=NULL);
 // system->init(Vector3(0.5,0.5,0.5),10,system->getSimulationData()->getParticleRadius());
}

//******************************************************************************
//******************************************************************************
void GLWidget::removeAll()
{
  if(system) delete(system);
  if(AHF) delete(AHF);
  init();
}

//******************************************************************************
//******************************************************************************
void GLWidget::draw() 
{
   if(display)
	display->display();
}
//******************************************************************************
//******************************************************************************
void GLWidget::animate()
{
 if(system!=NULL) system->update();
 if(AHF!=NULL) AHF->update();

 if(captureImage)
	captureImagesSequence();
}
//******************************************************************************
//******************************************************************************
void GLWidget::captureVideo(bool capture)
{
  captureImage = capture;
}
//******************************************************************************
//******************************************************************************
void GLWidget::captureImagesSequence()
{
   QPixmap image = renderPixmap (640,480,true);
   QString imageName = "../sortie/Video/image";
   QString str;
   str.setNum(frame_courante);
   if(frame_courante<10)
     str = "0000"+str;
   else {
    if(frame_courante<100)
        str = "000"+str;
    else { 
        if(frame_courante<1000)
            str = "00"+str;
        else {
            if(frame_courante<10000)
                 str = "0"+str;
            }
     }
   } 
   imageName+=str;
   imageName+=".jpg";
   image.save(imageName,"JPG");
   frame_courante++;
}
//******************************************************************************
//******************************************************************************
void GLWidget::displaySurface()
{
	if(system){
		if(typeid(*system)==typeid(SPHSystem) || typeid(*system)==typeid(PCI_SPHSystem)){
			if(surface==NULL)
				surface = new SurfaceSPH();
			surface->draw();
		}
	}
}
