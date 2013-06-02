#ifndef GLWIDGET_H
#define GLWIDGET_H


//************************************************************************/
//************************************************************************/

#include <QWidget>
#include <QtGui>
#include <QtOpenGL/QtOpenGL>
#include <math.h>
#include <QGLViewer/qglviewer.h>
#include <QGLViewer/vec.h>
#include <vector>

#include <System.h>
#include <SimpleSystem.h>
#include <CudaSystem.h>
#include <SphSystem.h>
#include <PciSphSystem.h>
#include <MSphSystem.h>

#include <SimulationData_CudaSystem.h>
#include <SimulationData_SimpleSystem.h>
#include <SimulationData_SPHSystem.h>
#include <SimulationData_PCI_SPHSystem.h>
#include <SimulationData_MSPHSystem.h>

#include <glDisplay.h>
#include <ObjectCollision.h>
#include <SurfaceSPH.h>

#include <AnimatedHeightField.h>
#include <AnimatedPeriodicHeightField.h>

//************************************************************************/
//************************************************************************/

using namespace qglviewer;
using namespace std;

//************************************************************************/
//************************************************************************/
 class GLWidget : public QGLViewer
 {
     Q_OBJECT

 public:

   //******************************************************************************
   // GETTERS ET SETTERS
   //******************************************************************************

     System*  getSystem();
     void     setSystem(System*);

     AnimatedHeightField *getAnimatedHF();
     void                 setAnimatedHF(AnimatedHeightField*);

     GLDisplay* getDisplay();
     void       setDisplay(GLDisplay*);

     SurfaceSPH* getSurface();

  //******************************************************************************
  // SURCHARGE DES FONCTIONS
  //******************************************************************************
     virtual void  init();
     virtual void  draw();
     virtual void  animate();
     virtual QSize minimumSizeHint() const;
     virtual QSize sizeHint() const;

  //******************************************************************************
  // FONCTIONS PROPRES
  //******************************************************************************
     void removeAll();
     void initSystem();

     void captureVideo(bool);
     void captureImagesSequence();

     void displaySurface();

  private :

     //******************************************************************************
     System *system;

     //******************************************************************************
     AnimatedHeightField *AHF;

     //******************************************************************************
     GLDisplay *display;

     //******************************************************************************
     SurfaceSPH *surface;

     //******************************************************************************
     unsigned int frame_courante;
     bool captureImage;

     //******************************************************************************
     void initRender();
 
 };

#endif
