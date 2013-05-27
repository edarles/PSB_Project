#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include <QtGui>
#include <QMainWindow>
#include <QtCore/QDir>
#include <QtGui>
#include <QMainWindow>
#include <glWidget.h>
#include <sceneWindow.h>

#include <SimpleSystem.h>
#include <ObjLoader.h>

//************************************************************************/
//************************************************************************/
class Window : public QWidget
{
    Q_OBJECT

//************************************************************************/
public:
    Window();

//************************************************************************/
public slots:

     //************************************************************************/
     // slots du menu Fichier
     void nouveau();
     void export_particles_txt();
     void export_particles_xml();
     void export_particles_Mitsuba();
     void import_particles_txt();
     void import_particles_xml();
     void export_surface_obj();
     void quit();

     //************************************************************************/
     // slots du menu Particles
     void simpleSystem();
     void particleInteractionSystem();
     void fluidSystem();
     void PCI_fluidSystem();

     //************************************************************************/
     // slots du menu Collision
     void createPlaneCollision();
     void createSphereCollision();
     void createBoxCollision();
     void createCylinderCollision();
     void createLinearHeightFieldCollision();
     void createGaussianHeightFieldCollision();
     void createPeriodicHeightFieldCollision();
     void createCombinedHeightFieldCollision();
//     void createAnimatedHeightFieldCollision(); //MATHIAS
     void loadFileCollision();
 
     //************************************************************************/
     // slots du menu Emitters
     void createEmitterBox();
     void createEmitterElipsoide();
     void createEmitterMesh();
     void createEmitterCylinder();
     void createEmitterSphere();
     void createEmitterGirly();

     //************************************************************************/
     // slots du menu Forces Exterieures
     void addForceConstante();
     void addForceTrochoide();

     //************************************************************************/
     // slots du menu Display
     void modeDisplay_Particles_POINT();
     void modeDisplay_Particles_SPHERE();
     void modeDisplay_Particles_SURFACE();
     void modeDisplay_Collisions_POINT();
     void modeDisplay_Collisions_LINES();
     void modeDisplay_Collisions_FILL();
     void modeDisplay_Collisions_normales(bool);
     void modeDisplay_DrawEmitters(bool);

     //************************************************************************/
     // slots du menu Animation
     void play();
     void stop();
     void captureFrames();

    
     //************************************************************************/
     // slots du menu Help
     void help();

private :

    //************************************************************************/
    GLWidget *glWidget;
    QMenuBar *menuBar;
    SceneWindow *sceneWindow;
    QHBoxLayout *layout2;
    QVBoxLayout *mainLayout;

    //************************************************************************/
    // MENU ET SOUS-MENU
    QMenu *menuFile;
    QMenu *menuExport;
    QMenu *menuImport;
    QMenu *menuExportParticles;
    QMenu *menuImportParticles;
    QMenu *menuExportSurface;
    QMenu *menuCollision;
    QMenu *menuHeightFieldCollision;
    QMenu *menuCreateCollision;
    QMenu *menuParticles;
    QMenu *menuEmitters;
    QMenu *menuForcesExt;
    QMenu *menuPeriodicForcesExt;
    QMenu *menuDisplay;
    QMenu *menuDisplay_Particles;
    QMenu *menuDisplay_Collisions;
    QMenu *menuAnimation;
    QMenu *menuHelp;
    QMenu *menuExtras; //MATHIAS
    QMenu *menuExtras_AnimatedHeightField;//MATHIAS
    QMenu *menuExtras_Floating;//MATHIAS

    //************************************************************************/
    // ACTION MENU FILE
    QAction *new_act;
    QAction *export_particles_txt_act;
    QAction *export_particles_xml_act;
    QAction *export_particles_Mitsuba_act;
    QAction *import_particles_txt_act;
    QAction *import_particles_xml_act;
    QAction *export_surface_obj_act;
    QAction *quit_act;

    //************************************************************************/
    // ACTION MENU COLLISION
    QAction *createPlaneCollision_act;
    QAction *createSphereCollision_act;
    QAction *createBoxCollision_act;
    QAction *createCylinderCollision_act;
    QAction *createLinearHeightFieldCollision_act;
    QAction *createGaussianHeightFieldCollision_act;
    QAction *createPeriodicHeightFieldCollision_act;
    QAction *createCombinedHeightFieldCollision_act;
    QAction *createAnimatedHeightFieldCollision_act; //MATHIAS
    QAction *loadFileCollision_act;
    QAction *createFloatingObject;

    //************************************************************************/
    // ACTION MENU PARTICLES
    QAction *simpleSystem_act;
    QAction *particleInteraction_act;
    QAction *fluidSystem_act;
    QAction *PCI_fluidSystem_act;

    //************************************************************************/
    // ACTION MENU EMITTERS
    QAction *createEmitterBox_act;
    QAction *createEmitterElipsoide_act;
    QAction *createEmitterMesh_act;
    QAction *createEmitterCylinder_act;
    QAction *createEmitterSphere_act;
    QAction *createEmitterGirly_act;

    //************************************************************************/
    // ACTION MENU FORCES EXTERIEURES
    QAction *addForceConstante_act;
    QAction *addForceTrochoide_act;

    //************************************************************************/
    // ACTION MENU DISPLAY
    QAction *modeDisplay_Particles_POINT_act;
    QAction *modeDisplay_Particles_SPHERE_act;
    QAction *modeDisplay_Particles_SURFACE_act;

    QAction *modeDisplay_Collisions_POINT_act;
    QAction *modeDisplay_Collisions_LINES_act;
    QAction *modeDisplay_Collisions_FILL_act;
    QAction *modeDisplay_Collisions_normales_act;

    QAction *modeDisplay_DrawEmitters_act;

    //************************************************************************/
    // ACTION MENU ANIMATION
    QAction *play_act;
    QAction *stop_act;
    QAction  *captureFrames_act;

    //************************************************************************/
    // Action Help
    QAction *help_act;
   
    //Action Extras
    QMenu *animated_heightfield;
    QAction *animated_heightfield_periodic;
    QAction *floating_object;
    
    std::string getOpenFileName(const QString & caption, const QStringList & filters,int * ind_filter);
   
    void alertBox(QString text, QMessageBox::Icon);

    QDir current_dir;

};
//************************************************************************/
//************************************************************************/
#endif

