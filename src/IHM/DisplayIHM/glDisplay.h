#ifndef GLDISPLAY_H
#define GLDISPLAY_H


//************************************************************************/
//************************************************************************/
#include <GL/gl.h>
#include <System.h>
//************************************************************************/
//************************************************************************/
#include <QWidget>
#include <QtGui>
#include <QtOpenGL/QtOpenGL>
#include <math.h>
#include <QGLViewer/qglviewer.h>
#include <QGLViewer/vec.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <SurfaceSPH.h>

using namespace qglviewer;

class GLWidget;

//************************************************************************/
//************************************************************************/
 class GLDisplay
 {
  public:

	GLDisplay(GLWidget*);
	GLDisplay(GLWidget*, System::ParticleDisplay modeParticles, GLenum modeFace, GLenum modeRasterization, bool drawNormales_ObjCollision);
	~GLDisplay();

	GLWidget *getGLWidget();
	void setGLWidget(GLWidget*);

	// GETTERS AND SETTERS //

	System::ParticleDisplay getModeParticles();
	void 		        setModeParticles(System::ParticleDisplay);

	GLenum getModeFace();
	GLenum getModeRasterization();
	bool   getDrawNormales_ObjCollision();
	bool   getDrawEmitters();
	bool   getDrawSurface();

	void setModeFace(GLenum);
	void setModeRasterization(GLenum);
	void setDrawNormales_ObjCollision(bool);
	void setDrawEmitters(bool);
	void setDrawSurface(bool);

	// PUBLIC METHOD TO DRAW
	void display();

	void displayObjectCollision(ObjectCollision *O);
	void setObjectIsCreate();
	void displayEmitter(Emitter* E);
        void displayParticlesByField(uint field);
	void displaySurface();

	Vector3 colorCollision;
	Vector3 colorParticles;
	Vector3 colorNormales;
	Vector3 colorEmitters;

  private :

	GLWidget *glWidget;
	// Mode to display particles
	System::ParticleDisplay modeParticles;
	// Mode to display collisions
	GLenum modeFace;
	GLenum modeRasterization;
	bool   drawNormales_ObjCollision;
	bool   drawEmitters;
	bool   drawSurface;
	SurfaceSPH *surfaceSPH;

	// PRIVATE METHODS TO DRAW
	void displayParticles();
	void displayCollisions();
	void displayEmitters();
	void displayAnimatedHeightField();

	// Draw object creation during configuration
	ObjectCollision *currentObj;

 };

#endif
