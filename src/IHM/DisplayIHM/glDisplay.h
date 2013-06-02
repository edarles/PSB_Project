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

using namespace qglviewer;
//using namespace qglviewer;
//using namespace std;

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

	void setModeFace(GLenum);
	void setModeRasterization(GLenum);
	void setDrawNormales_ObjCollision(bool);
	void setDrawEmitters(bool);

	// PUBLIC METHOD TO DRAW
	void display();

	void displayObjectCollision(ObjectCollision *O);
	void setObjectIsCreate();

	void displayEmitter(Emitter* E);

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

	// PRIVATE METHODS TO DRAW
	void displayParticles();
	void displayCollisions();
	void displayEmitters();
	void displayAnimatedHeightField();

	// Draw object creation during configuration
	ObjectCollision *currentObj;

 };

#endif
