#ifndef GLWIDGET_H
#define GLWIDGET_H


//************************************************************************/
//************************************************************************/

#include <QWidget>
#include <QtGui>
#include <math.h>
#include <QGLViewer/qglviewer.h>
#include <QGLViewer/vec.h>
#include <vector>

#include <System.h>
#include <SimpleSystem.h>
#include <CudaSystem.h>
#include <SphSystem.h>
#include <Sph2DSystem.h>
#include <WCSphSystem.h>
#include <PciSphSystem.h>
#include <MSphSystem.h>
#include <HSphSystem.h>
#include <SWSphSystem.h>
#include <HybridSphSystem.h>

#include <SimulationData_CudaSystem.h>
#include <SimulationData_SimpleSystem.h>
#include <SimulationData_SPHSystem.h>
#include <SimulationData_WCSPHSystem.h>
#include <SimulationData_SPHSystem2D.h>
#include <SimulationData_PCI_SPHSystem.h>
#include <SimulationData_MSPHSystem.h>
#include <SimulationData_HSPHSystem.h>
#include <SimulationData_SWSPHSystem.h>
#include <SimulationData_HybridSPHSystem.h>

#include <glDisplay.h>
#include <ObjectCollision.h>

#include <AnimatedHeightField.h>
#include <AnimatedPeriodicHeightField.h>
#include <AnimatedPeriodicHeightFieldCollision.h>
#include <SceneExporter_Mitsuba.h>
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
     void        setSurface(SurfaceSPH*);

     unsigned int getFrameCourante();

  //******************************************************************************
  // SURCHARGE DES FONCTIONS
  //******************************************************************************
     virtual void  init();
     virtual void  draw();
     virtual void  animate();
     virtual QSize minimumSizeHint() const;
     virtual QSize sizeHint() const;
     virtual void keyPressEvent( QKeyEvent *e );

  //******************************************************************************
  // FONCTIONS PROPRES
  //******************************************************************************
     void removeAll();
     void initSystem();
     void initConfig();

     void captureVideo(bool);
     void captureImagesSequence();
     void captureMitsubaSequence(const char* filename);

     void displayParticlesByField(uint field);

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
     bool displayByField;
     uint field;

     //******************************************************************************
     void initRender();
 
	//
	uint nbSoliton;
	float l;
 };

#endif
