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

#include <AnimatedHeightField.h>
#include <AnimatedPeriodicHeightField.h>
#include <System.h>
#include <SimpleSystem.h>
#include <CudaSystem.h>
#include <SphSystem.h>
#include <PciSphSystem.h>
#include <SimulationData_CudaSystem.h>
#include <SimulationData_SimpleSystem.h>
#include <SimulationData_SPHSystem.h>
#include <SimulationData_PCI_SPHSystem.h>
#include <glDisplay.h>
#include <ObjectCollision.h>
#include <SurfaceSPH.h>

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

     GLDisplay* getDisplay();
     void       setDisplay(GLDisplay*);

     SurfaceSPH* getSurface();
     
     AnimatedPeriodicHeightField* getAnimatedHF();
     void setAnimatedHF(AnimatedPeriodicHeightField * hf);

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
     GLDisplay *display;

     //******************************************************************************
     SurfaceSPH *surface;
     
      AnimatedPeriodicHeightField *animatedHF;

     //******************************************************************************
     unsigned int frame_courante;
     bool captureImage;

     //******************************************************************************
     void initRender();
 
 };

#endif
