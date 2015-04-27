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
    GLWidget* getGLWidget();
    void setGLWidget(GLWidget* glWidget);

//************************************************************************/
public slots:

     //************************************************************************/
     // slots du menu Fichier
     void nouveau();
     void export_particles_txt();
     void export_particles_txt_maya();
     void export_particles_xml();
     void export_particles_Mitsuba();
     void import_particles_txt();
     void import_particles_xml();
     void export_surface_obj();
     void export_scene_mitsuba();
     void quit();

     //************************************************************************/
     // slots du menu Particles
     void simpleSystem();
     void particleInteractionSystem();
     void fluidSystem();
     void PCI_fluidSystem();
     void WCSPHSystem();
     void Mixing_fluidSystem();
     void SWSPHSystem();
     void SPH2DSystem();
     void HybridSPHSystem();
     void HSPHSystem();

     //************************************************************************/
     // slots du menu Collision
     void createPlaneCollision();
     void createSphereCollision();
     void createBoxCollision();
     void createCylinderCollision();
     void createLinearHeightFieldCollision();
     void createGaussianHeightFieldCollision();
     void createPeriodicHeightFieldCollision();
     void createAnimatedPeriodicHeightFieldCollision();	
     void createCombinedHeightFieldCollision();
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
     void addForceSoliton();

     //************************************************************************/
     // slots du menu Display
     void modeDisplay_Particles_POINT();
     void modeDisplay_Particles_SPHERE();
     void modeDisplay_Particles_SURFACE();

     void modeDisplay_Particles_FIELD_DENSITY();
     void modeDisplay_Particles_FIELD_PRESSURE();
     void modeDisplay_Particles_FIELD_VISCOSITY();
     void modeDisplay_Particles_FIELD_TEMPERATURE();
     void modeDisplay_Particles_FIELD_CONCENTRATION();
     void modeDisplay_Particles_FIELD_MASS();
     void modeDisplay_Particles_FIELD_LEVEL();
     void modeDisplay_Particles_FIELD_STENSION();

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
     // slots du menu Extras
     void createAnimatedHeightField(); 
    
     //************************************************************************/
     // slots du menu Render
     void render();

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
    QMenu *menuExportScene;
    QMenu *menuCollision;
    QMenu *menuHeightFieldCollision;
    QMenu *menuCreateCollision;
    QMenu *menuParticles;
    QMenu *menuEmitters;
    QMenu *menuForcesExt;
    QMenu *menuPeriodicForcesExt;
    QMenu *menuDisplay;
    QMenu *menuRender;
    QMenu *menuDisplay_Particles;
    QMenu *menuDisplay_Collisions;
    //************************************************************************/
    // QMenu EXTRAS
    QMenu *menuExtras;
    QMenu *menuExtras_AnimatedHeightField;

    QMenu *menuAnimation;
    QMenu *menuHelp;

    //************************************************************************/
    // ACTION MENU FILE
    QAction *new_act;
    QAction *export_particles_txt_act;
    QAction *export_particles_txt_maya_act;
    QAction *export_particles_xml_act;
    QAction *export_particles_Mitsuba_act;
    QAction *import_particles_txt_act;
    QAction *import_particles_xml_act;
    QAction *export_surface_obj_act;
    QAction *export_scene_mitsuba_act;
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
    QAction *createAnimatedPeriodicHeightFieldCollision_act;
    QAction *createCombinedHeightFieldCollision_act;
    QAction *loadFileCollision_act;

    //************************************************************************/
    // ACTION MENU PARTICLES
    QAction *simpleSystem_act;
    QAction *particleInteraction_act;
    QAction *fluidSystem_act;
    QAction *SPH2DSystem_act;
    QAction *PCI_fluidSystem_act;
    QAction *WCSPHSystem_act;
    QAction *Mixing_fluidSystem_act;
    QAction *Hierarchical_fluidSystem_act;
    QAction *SWSPHSystem_act;
    QAction *HybridSPHSystem_act;

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
    QAction *addForceSoliton_act;

    //************************************************************************/
    // ACTION MENU DISPLAY
    QAction *modeDisplay_Particles_POINT_act;
    QAction *modeDisplay_Particles_SPHERE_act;
    QAction *modeDisplay_Particles_SURFACE_act;
    QMenu   *menu_modeDisplay_Particles_FIELD;
    QAction *modeDisplay_Particles_FIELD_DENSITY_act;
    QAction *modeDisplay_Particles_FIELD_PRESSURE_act;
    QAction *modeDisplay_Particles_FIELD_VISCOSITY_act;
    QAction *modeDisplay_Particles_FIELD_TEMPERATURE_act;
    QAction *modeDisplay_Particles_FIELD_CONCENTRATION_act;
    QAction *modeDisplay_Particles_FIELD_MASS_act;
    QAction *modeDisplay_Particles_FIELD_LEVEL_act;
    QAction *modeDisplay_Particles_FIELD_STENSION_act;

    QAction *modeDisplay_Collisions_POINT_act;
    QAction *modeDisplay_Collisions_LINES_act;
    QAction *modeDisplay_Collisions_FILL_act;
    QAction *modeDisplay_Collisions_normales_act;

    QAction *modeDisplay_DrawEmitters_act;

    //************************************************************************/
    // ACTION MENU EXTRAS
    QAction *createAnimatedHeightField_act;

    //************************************************************************/
    // ACTION MENU ANIMATION
    QAction *play_act;
    QAction *stop_act;
    QAction  *captureFrames_act;

    //************************************************************************/
    // ACTION MENU RENDERER
    QAction *renderingFrame_act;

    //************************************************************************/
    // Action Help
    QAction *help_act;
   
    std::string getOpenFileName(const QString & caption, const QStringList & filters,int * ind_filter);
   
    void alertBox(QString text, QMessageBox::Icon);

    QDir current_dir;

};
//************************************************************************/
//************************************************************************/
#endif

