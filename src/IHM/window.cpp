#include <window.h>

#include <windowConfiguration_SimpleSystem.h>
#include <windowConfiguration_CudaSystem.h>
#include <windowConfiguration_SPHSystem.h>
#include <windowConfiguration_SPH2DSystem.h>
#include <windowConfiguration_WCSPHSystem.h>
#include <windowConfiguration_PCI_SPHSystem.h>
#include <windowConfiguration_MSPHSystem.h>
#include <windowConfiguration_HSPHSystem.h>
#include <windowConfiguration_SWSPHSystem.h>
#include <windowConfiguration_HybridSPHSystem.h>

#include <windowConfiguration_Sphere.h>
#include <windowConfiguration_Box.h>
#include <windowConfiguration_Mesh.h>
#include <windowConfiguration_Plan.h>
#include <windowConfiguration_Cylinder.h>
#include <windowConfiguration_LinearHeightField.h>
#include <windowConfiguration_GaussianHeightField.h>
#include <windowConfiguration_PeriodicHeightField.h>
#include <windowConfiguration_AnimatedPeriodicHeightField.h>
#include <windowConfiguration_CombinedHeightField.h>

#include <windowConfiguration_Emitter_Box.h>
#include <windowConfiguration_Emitter_Elipsoide.h>
#include <windowConfiguration_Emitter_Mesh.h>
#include <windowConfiguration_Emitter_Cylinder.h>
#include <windowConfiguration_Emitter_Sphere.h>
#include <windowConfiguration_Emitter_Girly.h>

#include <windowConfiguration_ForceExt_Trochoide.h>
#include <windowConfiguration_ForceExt_Constante.h>
#include <windowConfiguration_ForceExt_Soliton.h>

#include <windowConfiguration_AnimatedHeightField.h>

#include <ParticleExporter_Txt.h>
#include <ParticleExporter_XML.h>
#include <ParticleExporter_Mitsuba.h>
#include <SceneExporter_Mitsuba.h>

#include <ParticleLoader_Txt.h>
#include <ParticleLoader_XML.h>

//#include <RLite.h>
//************************************************************************/
//************************************************************************/
Window::Window():QWidget()
{
    glWidget = new GLWidget();
    menuBar = new QMenuBar();
    sceneWindow = NULL;

    // MENU FILE
    menuFile = new QMenu("&File");

    new_act = new QAction(tr("&New"),this);
    new_act->setShortcut(tr("n"));
    connect(new_act,SIGNAL(triggered()), this, SLOT(nouveau()));

    menuImport = new QMenu ("&Import",this);

    menuImportParticles = new QMenu(tr("&Particles"),this);
    import_particles_txt_act = new QAction(tr("&Txt"),this);
    connect(import_particles_txt_act,SIGNAL(triggered()), this, SLOT(import_particles_txt()));
    menuImportParticles->addAction(import_particles_txt_act);
    import_particles_xml_act = new QAction(tr("&XML"),this);
    connect(import_particles_xml_act,SIGNAL(triggered()), this, SLOT(import_particles_xml()));
    menuImportParticles->addAction(import_particles_xml_act);
    menuImport->addMenu(menuImportParticles);

    menuExport = new QMenu ("&Export",this);

    menuExportParticles = new QMenu(tr("&Particles"),this);
    export_particles_txt_act = new QAction(tr("&Txt"),this);
    connect(export_particles_txt_act,SIGNAL(triggered()), this, SLOT(export_particles_txt()));
    menuExportParticles->addAction(export_particles_txt_act);
    export_particles_txt_maya_act = new QAction(tr("&Txt (To convert Maya PDC)"),this);
    connect(export_particles_txt_maya_act,SIGNAL(triggered()), this, SLOT(export_particles_txt_maya()));
    menuExportParticles->addAction(export_particles_txt_maya_act);
    export_particles_xml_act = new QAction(tr("&XML"),this);
    connect(export_particles_xml_act,SIGNAL(triggered()), this, SLOT(export_particles_xml()));
    menuExportParticles->addAction(export_particles_xml_act);
    export_particles_Mitsuba_act = new QAction(tr("&Mitsuba (vol)"),this);
    connect(export_particles_Mitsuba_act,SIGNAL(triggered()), this, SLOT(export_particles_Mitsuba()));
    menuExportParticles->addAction(export_particles_Mitsuba_act);
    menuExport->addMenu(menuExportParticles);

    menuExportSurface = new QMenu(tr("FluidSurface"),this);
    export_surface_obj_act = new QAction(tr("&OBJ"),this);
    connect(export_surface_obj_act,SIGNAL(triggered()), this, SLOT(export_surface_obj()));
    menuExportSurface->addAction(export_surface_obj_act);
    menuExport->addMenu(menuExportSurface);

    menuExportScene = new QMenu(tr("Scene"),this);
    export_scene_mitsuba_act = new QAction(tr("&Mitsuba"),this);
    connect(export_scene_mitsuba_act,SIGNAL(triggered()), this, SLOT(export_scene_mitsuba()));
    menuExportScene->addAction(export_scene_mitsuba_act);
    menuExport->addMenu(menuExportScene);

    quit_act = new QAction(tr("&Quit"), this);
    quit_act->setShortcut(tr("Q"));
    connect(quit_act, SIGNAL(triggered()), this, SLOT(quit()));

    menuFile->addAction(new_act);
    menuFile->addSeparator();
    menuFile->addMenu(menuImport);
    menuFile->addSeparator();
    menuFile->addMenu(menuExport);
    menuFile->addSeparator();
    menuFile->addAction(quit_act);

    // MENU PARTICLES
    menuParticles = new QMenu("&System");
/*
    simpleSystem_act = new QAction(tr("&Simple System"),this);
    connect(simpleSystem_act, SIGNAL(triggered()), this, SLOT(simpleSystem()));
    menuParticles->addAction(simpleSystem_act);

    particleInteraction_act = new QAction(tr("&CUDA System"),this);
    connect(particleInteraction_act, SIGNAL(triggered()), this, SLOT(particleInteractionSystem()));
    menuParticles->addAction(particleInteraction_act);
*/
    QMenu *menuSystem3D = new QMenu("3D");
    fluidSystem_act = new QAction(tr("SPH3D"),this);
    connect(fluidSystem_act, SIGNAL(triggered()), this, SLOT(fluidSystem()));
    menuSystem3D->addAction(fluidSystem_act);
    QMenu *menuSystem3DWC = new QMenu("Weakly Compressed");
    WCSPHSystem_act = new QAction(tr("WC-SPH"),this);
    connect(WCSPHSystem_act, SIGNAL(triggered()), this, SLOT(WCSPHSystem()));
    menuSystem3DWC->addAction(WCSPHSystem_act);
    PCI_fluidSystem_act = new QAction(tr("PCI-SPH"),this);
    connect(PCI_fluidSystem_act, SIGNAL(triggered()), this, SLOT(PCI_fluidSystem()));
    menuSystem3DWC->addAction(PCI_fluidSystem_act);
    menuSystem3D->addMenu(menuSystem3DWC);
    Mixing_fluidSystem_act = new QAction(tr("Mixing-SPH"),this);
    connect(Mixing_fluidSystem_act, SIGNAL(triggered()), this, SLOT(Mixing_fluidSystem()));
    menuSystem3D->addAction(Mixing_fluidSystem_act);
    Hierarchical_fluidSystem_act = new QAction(tr("HSPH"),this);
    connect(Hierarchical_fluidSystem_act, SIGNAL(triggered()), this, SLOT(HSPHSystem()));
    menuSystem3D->addAction(Hierarchical_fluidSystem_act);
    HybridSPHSystem_act = new QAction(tr("Hybrid-SPH/SWSPH"),this);
    connect(HybridSPHSystem_act, SIGNAL(triggered()), this, SLOT(HybridSPHSystem()));
    menuSystem3D->addAction(HybridSPHSystem_act);
    menuParticles->addMenu(menuSystem3D);

    QMenu *menuSystem2D = new QMenu("2D");
    SPH2DSystem_act = new QAction(tr("SPH2D"),this);
    connect(SPH2DSystem_act, SIGNAL(triggered()), this, SLOT(SPH2DSystem()));
    menuSystem2D->addAction(SPH2DSystem_act);
    menuParticles->addMenu(menuSystem2D);

    QMenu *menuSystem2DH = new QMenu("2D+H");
    SWSPHSystem_act = new QAction(tr("SW-SPH"),this);
    connect(SWSPHSystem_act, SIGNAL(triggered()), this, SLOT(SWSPHSystem()));
    menuSystem2DH->addAction(SWSPHSystem_act);
    menuParticles->addMenu(menuSystem2DH);

    // MENU COLLISION
    menuCollision = new QMenu("&Collision");

    // SOUS MENU CREATE
    menuCreateCollision = new QMenu("&Create");

    createPlaneCollision_act = new QAction(tr("Plan"), this);
    connect(createPlaneCollision_act,SIGNAL(triggered()), this, SLOT(createPlaneCollision()));

    createSphereCollision_act = new QAction(tr("Sphere"),this);
    connect(createSphereCollision_act,SIGNAL(triggered()), this, SLOT(createSphereCollision()));

    createBoxCollision_act = new QAction(tr("Box"),this);
    connect(createBoxCollision_act,SIGNAL(triggered()), this, SLOT(createBoxCollision()));
    
    createCylinderCollision_act = new QAction(tr("Cylinder"),this);
    connect(createCylinderCollision_act,SIGNAL(triggered()), this, SLOT(createCylinderCollision()));

    menuHeightFieldCollision = new QMenu("&Height Field");

    createLinearHeightFieldCollision_act = new QAction(tr("Linear"), this);
    connect(createLinearHeightFieldCollision_act,SIGNAL(triggered()), this, SLOT(createLinearHeightFieldCollision()));
    menuHeightFieldCollision->addAction(createLinearHeightFieldCollision_act);

    createGaussianHeightFieldCollision_act = new QAction(tr("Gaussian"), this);
    connect(createGaussianHeightFieldCollision_act,SIGNAL(triggered()), this, SLOT(createGaussianHeightFieldCollision()));
    menuHeightFieldCollision->addAction(createGaussianHeightFieldCollision_act);

    createPeriodicHeightFieldCollision_act = new QAction(tr("Periodic"), this);
    connect(createPeriodicHeightFieldCollision_act,SIGNAL(triggered()), this, SLOT(createPeriodicHeightFieldCollision()));
    menuHeightFieldCollision->addAction(createPeriodicHeightFieldCollision_act);

    createAnimatedPeriodicHeightFieldCollision_act = new QAction(tr("AnimatedPeriodic"), this);
    connect(createAnimatedPeriodicHeightFieldCollision_act,SIGNAL(triggered()), this, SLOT(createAnimatedPeriodicHeightFieldCollision()));
    menuHeightFieldCollision->addAction(createAnimatedPeriodicHeightFieldCollision_act);

    createCombinedHeightFieldCollision_act = new QAction(tr("Combined"), this);
    connect(createCombinedHeightFieldCollision_act,SIGNAL(triggered()), this, SLOT(createCombinedHeightFieldCollision()));
    menuHeightFieldCollision->addAction(createCombinedHeightFieldCollision_act);

    menuCreateCollision->addAction(createPlaneCollision_act);
    menuCreateCollision->addAction(createSphereCollision_act);
    menuCreateCollision->addAction(createBoxCollision_act);
    menuCreateCollision->addAction(createCylinderCollision_act);
    menuCreateCollision->addMenu(menuHeightFieldCollision);

    // FIN SOUS MENU CREATE
    loadFileCollision_act = new QAction(tr("Load file"),this);
    connect(loadFileCollision_act,SIGNAL(triggered()), this, SLOT(loadFileCollision()));

    menuCollision->addMenu(menuCreateCollision);
    menuCollision->addAction(loadFileCollision_act);

    // MENU EMITTERS
    menuEmitters = new QMenu("&Emitters");

    createEmitterBox_act = new QAction(tr("Box"),this);
    connect(createEmitterBox_act, SIGNAL(triggered()), this, SLOT(createEmitterBox()));
    menuEmitters->addAction(createEmitterBox_act);

    createEmitterElipsoide_act = new QAction(tr("Elipsoide"),this);
    connect(createEmitterElipsoide_act, SIGNAL(triggered()), this, SLOT(createEmitterElipsoide()));
    menuEmitters->addAction(createEmitterElipsoide_act);

    createEmitterCylinder_act = new QAction(tr("Cylinder"),this);
    connect(createEmitterCylinder_act, SIGNAL(triggered()), this, SLOT(createEmitterCylinder()));
    menuEmitters->addAction(createEmitterCylinder_act);

    createEmitterSphere_act = new QAction(tr("Sphere"),this);
    connect(createEmitterSphere_act, SIGNAL(triggered()), this, SLOT(createEmitterSphere()));
    menuEmitters->addAction(createEmitterSphere_act);

    createEmitterGirly_act = new QAction(tr("Girly :-)"),this);
    connect(createEmitterGirly_act, SIGNAL(triggered()), this, SLOT(createEmitterGirly()));
    menuEmitters->addAction(createEmitterGirly_act);

    createEmitterMesh_act = new QAction(tr("Emit from mesh"),this);
    connect(createEmitterMesh_act, SIGNAL(triggered()), this, SLOT(createEmitterMesh()));
    menuEmitters->addAction(createEmitterMesh_act);

    // MENU FORCES EXTERNES
    menuForcesExt = new QMenu("&Forces externes");

    addForceConstante_act = new QAction(tr("Constante"),this);
    connect(addForceConstante_act, SIGNAL(triggered()), this, SLOT(addForceConstante()));
    menuForcesExt->addAction(addForceConstante_act);

    addForceSoliton_act = new QAction(tr("Soliton"),this);
    connect(addForceSoliton_act, SIGNAL(triggered()), this, SLOT(addForceSoliton()));
    menuForcesExt->addAction(addForceSoliton_act);

    menuPeriodicForcesExt = new QMenu("Periodic",this);
    addForceTrochoide_act = new QAction(tr("Trochoidal"),this);
    connect(addForceTrochoide_act, SIGNAL(triggered()), this, SLOT(addForceTrochoide()));
    menuPeriodicForcesExt->addAction(addForceTrochoide_act);
    menuForcesExt->addMenu(menuPeriodicForcesExt);

    // MENU DISPLAY
    menuDisplay = new QMenu("Display");
  
    menuDisplay_Particles = new QMenu("Particles");
   
    modeDisplay_Particles_POINT_act = new QAction(tr("Points"),this);
    connect(modeDisplay_Particles_POINT_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_POINT()));
    menuDisplay_Particles->addAction(modeDisplay_Particles_POINT_act);
    modeDisplay_Particles_SPHERE_act = new QAction(tr("Spheres"),this);
    connect(modeDisplay_Particles_SPHERE_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_SPHERE()));
    menuDisplay_Particles->addAction(modeDisplay_Particles_SPHERE_act);
    modeDisplay_Particles_SURFACE_act = new QAction(tr("Surface"),this);
    connect(modeDisplay_Particles_SURFACE_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_SURFACE()));
    menuDisplay_Particles->addAction(modeDisplay_Particles_SURFACE_act);

    menu_modeDisplay_Particles_FIELD = new QMenu("By field");
    modeDisplay_Particles_FIELD_DENSITY_act = new QAction(tr("Density"),this);
    connect(modeDisplay_Particles_FIELD_DENSITY_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_FIELD_DENSITY()));
    menu_modeDisplay_Particles_FIELD->addAction(modeDisplay_Particles_FIELD_DENSITY_act);
    modeDisplay_Particles_FIELD_PRESSURE_act = new QAction(tr("Pressure"),this);
    connect(modeDisplay_Particles_FIELD_PRESSURE_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_FIELD_PRESSURE()));
    menu_modeDisplay_Particles_FIELD->addAction(modeDisplay_Particles_FIELD_PRESSURE_act);
    modeDisplay_Particles_FIELD_VISCOSITY_act = new QAction(tr("Viscosity"),this);
    connect(modeDisplay_Particles_FIELD_VISCOSITY_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_FIELD_VISCOSITY()));
    menu_modeDisplay_Particles_FIELD->addAction(modeDisplay_Particles_FIELD_VISCOSITY_act);
    modeDisplay_Particles_FIELD_STENSION_act = new QAction(tr("Surface Tension"),this);
    connect(modeDisplay_Particles_FIELD_STENSION_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_FIELD_STENSION()));
    menu_modeDisplay_Particles_FIELD->addAction(modeDisplay_Particles_FIELD_STENSION_act);
    modeDisplay_Particles_FIELD_TEMPERATURE_act = new QAction(tr("Temperature"),this);
    connect(modeDisplay_Particles_FIELD_TEMPERATURE_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_FIELD_TEMPERATURE()));
    menu_modeDisplay_Particles_FIELD->addAction(modeDisplay_Particles_FIELD_TEMPERATURE_act);
    modeDisplay_Particles_FIELD_CONCENTRATION_act = new QAction(tr("Concentration"),this);
    connect(modeDisplay_Particles_FIELD_CONCENTRATION_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_FIELD_CONCENTRATION()));
    menu_modeDisplay_Particles_FIELD->addAction(modeDisplay_Particles_FIELD_CONCENTRATION_act);
    modeDisplay_Particles_FIELD_MASS_act = new QAction(tr("Mass"),this);
    connect(modeDisplay_Particles_FIELD_MASS_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_FIELD_MASS()));
    menu_modeDisplay_Particles_FIELD->addAction(modeDisplay_Particles_FIELD_MASS_act);
    modeDisplay_Particles_FIELD_LEVEL_act = new QAction(tr("Level"),this);
    connect(modeDisplay_Particles_FIELD_LEVEL_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Particles_FIELD_LEVEL()));
    menu_modeDisplay_Particles_FIELD->addAction(modeDisplay_Particles_FIELD_LEVEL_act);
    menuDisplay_Particles->addMenu(menu_modeDisplay_Particles_FIELD);

    menuDisplay_Collisions = new QMenu("Collisions");
    modeDisplay_Collisions_POINT_act = new QAction(tr("Points"),this);
    connect(modeDisplay_Collisions_POINT_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Collisions_POINT()));
    menuDisplay_Collisions->addAction(modeDisplay_Collisions_POINT_act);
    modeDisplay_Collisions_LINES_act = new QAction(tr("Lines"),this);
    connect(modeDisplay_Collisions_LINES_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Collisions_LINES()));
    menuDisplay_Collisions->addAction(modeDisplay_Collisions_LINES_act);
    modeDisplay_Collisions_FILL_act = new QAction(tr("Fill"),this);
    connect(modeDisplay_Collisions_FILL_act, SIGNAL(triggered()), this, SLOT(modeDisplay_Collisions_FILL()));
    menuDisplay_Collisions->addAction(modeDisplay_Collisions_FILL_act);
    modeDisplay_Collisions_normales_act = new QAction(tr("Normale"),this);
    modeDisplay_Collisions_normales_act->setCheckable(true);
    connect(modeDisplay_Collisions_normales_act, SIGNAL(toggled(bool)), this, SLOT(modeDisplay_Collisions_normales(bool)));
    menuDisplay_Collisions->addAction(modeDisplay_Collisions_normales_act);

    menuDisplay->addMenu(menuDisplay_Particles);
    menuDisplay->addMenu(menuDisplay_Collisions);

    modeDisplay_DrawEmitters_act = new QAction(tr("Emitters"),this);
    modeDisplay_DrawEmitters_act->setCheckable(true);
    modeDisplay_DrawEmitters_act->setChecked(true);
    connect(modeDisplay_DrawEmitters_act, SIGNAL(toggled(bool)), this, SLOT(modeDisplay_DrawEmitters(bool)));
    menuDisplay->addAction(modeDisplay_DrawEmitters_act);

    // MENU EXTRAS
    menuExtras = new QMenu("&Extras");
    menuExtras_AnimatedHeightField = new QMenu("&AnimatedHeightField");
    createAnimatedHeightField_act = new QAction(tr("&Periodic"),this);
    connect(createAnimatedHeightField_act , SIGNAL(triggered()), this, SLOT(createAnimatedHeightField()));
    menuExtras_AnimatedHeightField->addAction(createAnimatedHeightField_act);
    menuExtras->addMenu(menuExtras_AnimatedHeightField);

    // MENU ANIMATION
    menuAnimation = new QMenu("&Animation");   
    play_act = new QAction(tr("Play"),this);
    connect(play_act, SIGNAL(triggered()), this, SLOT(play()));
    stop_act = new QAction(tr("Stop"),this);
    connect(stop_act, SIGNAL(triggered()), this, SLOT(stop()));
    captureFrames_act = new QAction(tr("Capture Video"),this);
    connect(captureFrames_act, SIGNAL(triggered()), this, SLOT(captureFrames()));
    menuAnimation->addAction(play_act);
    menuAnimation->addAction(stop_act);
    menuAnimation->addSeparator();
    menuAnimation->addAction(captureFrames_act);
	
    // MENU RENDERER
    menuRender = new QMenu("&Render");
    renderingFrame_act = new QAction(tr("Current frame"),this);
    connect(renderingFrame_act, SIGNAL(triggered()), this, SLOT(render()));
    menuRender->addAction(renderingFrame_act);

   // MENU AIDE
    menuHelp = new QMenu("&Help");
    help_act = new QAction(tr("&Display"),this);
    help_act->setShortcut(tr("H"));
    connect(help_act,SIGNAL(triggered()), this, SLOT(help()));
    menuHelp->addAction(help_act);
   
   // AJOUT DES MENUS AU MENU BAR DE LA FENETRE
    menuBar->addMenu(menuFile);
    menuBar->addMenu(menuParticles);
    menuBar->addMenu(menuEmitters);
    menuBar->addMenu(menuCollision);
    menuBar->addMenu(menuForcesExt);
    menuBar->addMenu(menuDisplay);
    menuBar->addMenu(menuAnimation);
    menuBar->addMenu(menuRender);
    menuBar->addMenu(menuExtras);
    menuBar->addMenu(menuHelp);

    mainLayout = new QVBoxLayout();
    mainLayout->addWidget(menuBar);
    layout2 = new QHBoxLayout();
    layout2->addWidget(glWidget);
    mainLayout->addLayout(layout2);
    setLayout(mainLayout);

    setWindowTitle(tr("Particles Simulator Benchmarking (PSB)"));

}
//************************************************************************/
//************************************************************************/
GLWidget* Window::getGLWidget()
{
	return glWidget;
}
//************************************************************************/
//************************************************************************/
void Window::setGLWidget(GLWidget* G)
{
	 this->glWidget = G;
}
//************************************************************************/
//************************************************************************/
// SLOTS MENU FILE
//************************************************************************/
//************************************************************************/
void Window::nouveau()
{
	glWidget->removeAll();

	if(mainLayout!=NULL){
		mainLayout->removeItem(layout2);
		setLayout(mainLayout);
		/*qDeleteAll(layout2->children());
		qDeleteAll(mainLayout->children());
		//delete(layout2);
    		layout2 = new QHBoxLayout();
    		layout2->addWidget(glWidget);
    		mainLayout->addLayout(layout2);
    		setLayout(mainLayout);*/
	}
}
/************************************************************************/
void Window::import_particles_txt()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
 		current_dir = "Ressources/Particles/";
 	        QStringList type_load; 
		type_load << "txt file (*.txt)";
 	        std::string filename = getOpenFileName("Specify a txt file", type_load,0);
		if (filename != ""){
			ParticleLoader_Txt loader;
    			vector<Particle*> particles = loader.load(S,filename.c_str());
			S->init(particles);
		}
 }
 else {
	        alertBox("Pas de système à exporter !!",QMessageBox::Critical);
 }
}
/************************************************************************/
void Window::import_particles_xml()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
 		current_dir = "Ressources/Particles/";
		QStringList type_load;
 	        type_load << "xml file (*.xml)";
 	        std::string filename = getOpenFileName("Specify a xml file", type_load,0);
		if (filename != ""){
			ParticleLoader_XML loader;
    			vector<Particle*> particles = loader.load(S,filename.c_str());
			S->init(particles);
		}
 }
 else {
	        alertBox("Pas de système à exporter !!",QMessageBox::Critical);
 }
}
/************************************************************************/
void Window::export_particles_txt()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
        	current_dir = "sortie/Particles/Txt";
 		QString filename = QFileDialog::getSaveFileName(this, tr("Save particles"), "*.txt");
		ParticleExporter_Txt exporter;
    		exporter._export(filename.toStdString().c_str(),S);
 }
 else {
	        alertBox("Pas de système à exporter !!",QMessageBox::Critical);
 }
}
/************************************************************************/
void Window::export_particles_txt_maya()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	printf("EXPORT SIMULATION (Txt format to convert into PDC Maya format)\n");
        current_dir = "sortie/Particles/Txt/toPDC/";
        QString dir = "../sortie/Particles/Txt/toPDC/";
	for(unsigned int i=0;i<1000;i++){
		printf("..%d..",glWidget->getFrameCourante());
		glWidget->animate();
		ParticleExporter_Txt exporter;
    		exporter._exportToMaya(glWidget->getFrameCourante(),dir.toStdString().c_str(),S);
	}
 }
 else {
	        alertBox("Pas de système à exporter !!",QMessageBox::Critical);
 }
}
/************************************************************************/
void Window::export_particles_xml()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
        	current_dir = "sortie/Particles/XML/";
 		QString filename = QFileDialog::getSaveFileName(this, tr("Save particles"), "*.xml");
		ParticleExporter_XML exporter;
    		exporter._export(filename.toStdString().c_str(),S);
 }
 else {
	        alertBox("Pas de système à exporter !!",QMessageBox::Critical);
 }
}
/************************************************************************/
void Window::export_particles_Mitsuba()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	if(typeid(*S)==typeid(MSPHSystem)){
		MSPHSystem* SPH = (MSPHSystem*) S;
        	current_dir = "sortie/";
 		QString filenameDensity = QFileDialog::getSaveFileName(this, tr("Save density vol"), "*.vol");
		QString filenameAlbedo = QFileDialog::getSaveFileName(this, tr("Save albedo vol"), "*.vol");
		QString filenameSigmaS = QFileDialog::getSaveFileName(this, tr("Save sigmaS vol"), "*.vol");
		QString filenameSigmaT = QFileDialog::getSaveFileName(this, tr("Save sigmaT vol"), "*.vol");
		ParticleExporter_Mitsuba exp;
		exp._exportData(filenameDensity.toStdString().c_str(),filenameAlbedo.toStdString().c_str(),
				filenameSigmaS.toStdString().c_str(), filenameSigmaT.toStdString().c_str(), SPH);
	}
	else
		 alertBox("Uniquement pour des MSPHSystem !!",QMessageBox::Critical);
 }
 else {
	        alertBox("Pas de système à exporter !!",QMessageBox::Critical);
 }
}

/************************************************************************/
void Window::export_surface_obj()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
        	current_dir = "sortie/";
 		QString filename = QFileDialog::getSaveFileName(this, tr("Save surface OBJ format"), "*.obj");
		glWidget->getSurface()->exportOBJ(filename.toStdString().c_str());
 }
 else {
	        alertBox("Pas de surface à exporter !!",QMessageBox::Critical);
 }
}
/************************************************************************/
void Window::export_scene_mitsuba()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
        	current_dir = "sortie/";
 		QString filename = QFileDialog::getSaveFileName(this, tr("Save scene (mitsuba format)"), "*.xml");
		SceneExporter_Mitsuba exporter;
    		exporter._export(filename.toStdString().c_str(),S,Vector3(1.0,0.0,0.0));
 }
 else {
	        alertBox("Pas de système à exporter !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::quit()
{
	this->close() ;
}
//************************************************************************/
// SLOTS MENU PARTICLES
/************************************************************************/
//************************************************************************/
void Window::simpleSystem()
{
   WindowConfiguration_SimpleSystem *windowConfig = new WindowConfiguration_SimpleSystem(this,mainLayout, layout2, glWidget);
   windowConfig->show();
}
//************************************************************************/
void Window::particleInteractionSystem()
{
   WindowConfiguration_CudaSystem *windowConfig = new WindowConfiguration_CudaSystem(glWidget);
   windowConfig->show();
}
//************************************************************************/
void Window::SPH2DSystem()
{
   WindowConfiguration_SPH2DSystem *windowConfig = new WindowConfiguration_SPH2DSystem(glWidget);
   windowConfig->show();
}
//************************************************************************/
void Window::fluidSystem()
{
   WindowConfiguration_SPHSystem *windowConfig = new WindowConfiguration_SPHSystem(glWidget);
   windowConfig->show();
}
//************************************************************************/
void Window::WCSPHSystem()
{
   WindowConfiguration_WCSPHSystem *windowConfig = new WindowConfiguration_WCSPHSystem(glWidget);
   windowConfig->show();
}
//************************************************************************/
void Window::PCI_fluidSystem()
{
   WindowConfiguration_PCI_SPHSystem *windowConfig = new WindowConfiguration_PCI_SPHSystem(glWidget);
   windowConfig->show();
}
//************************************************************************/
void Window::Mixing_fluidSystem()
{
   WindowConfiguration_MSPHSystem *windowConfig = new WindowConfiguration_MSPHSystem(glWidget);
   windowConfig->show();
}
//************************************************************************/
void Window::HSPHSystem()
{
   WindowConfiguration_HSPHSystem *windowConfig = new WindowConfiguration_HSPHSystem(glWidget);
   windowConfig->show();
}
//************************************************************************/
void Window::SWSPHSystem()
{
   WindowConfiguration_SWSPHSystem *windowConfig = new WindowConfiguration_SWSPHSystem(glWidget);
   windowConfig->show();
}
//************************************************************************/
void Window::HybridSPHSystem()
{
   WindowConfiguration_HybridSPHSystem *windowConfig = new WindowConfiguration_HybridSPHSystem(glWidget);
   windowConfig->show();
}

//************************************************************************/
// SLOTS MENU COLLISION
/************************************************************************/
//************************************************************************/
void Window::createPlaneCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
        WindowConfiguration_Plan *windowConfig = new WindowConfiguration_Plan(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createSphereCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
        WindowConfiguration_Sphere *windowConfig = new WindowConfiguration_Sphere(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createBoxCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_Box *windowConfig = new WindowConfiguration_Box(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createCylinderCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_Cylinder *windowConfig = new WindowConfiguration_Cylinder(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createLinearHeightFieldCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_LinearHeightField *windowConfig = new WindowConfiguration_LinearHeightField(NULL,glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createGaussianHeightFieldCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_GaussianHeightField *windowConfig = new WindowConfiguration_GaussianHeightField(NULL,glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createPeriodicHeightFieldCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_PeriodicHeightField *windowConfig = new WindowConfiguration_PeriodicHeightField(NULL,glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createAnimatedPeriodicHeightFieldCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_AnimatedPeriodicHeightField *windowConfig = new WindowConfiguration_AnimatedPeriodicHeightField(NULL,glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createCombinedHeightFieldCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_CombinedHeightField *windowConfig = new WindowConfiguration_CombinedHeightField(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::loadFileCollision()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
 	WindowConfiguration_Mesh *windowConfig = new WindowConfiguration_Mesh(glWidget);
 	windowConfig->show();	
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
// SLOTS MENU EMITTERS
/************************************************************************/
//************************************************************************/
void Window::createEmitterBox()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_Emitter_Box *windowConfig = new WindowConfiguration_Emitter_Box(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createEmitterElipsoide()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_Emitter_Elipsoide *windowConfig = new WindowConfiguration_Emitter_Elipsoide(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createEmitterMesh()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_Emitter_Mesh *windowConfig = new WindowConfiguration_Emitter_Mesh(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createEmitterCylinder()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_Emitter_Cylinder *windowConfig = new WindowConfiguration_Emitter_Cylinder(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createEmitterSphere()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_Emitter_Sphere *windowConfig = new WindowConfiguration_Emitter_Sphere(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::createEmitterGirly()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
	WindowConfiguration_Emitter_Girly *windowConfig = new WindowConfiguration_Emitter_Girly(glWidget);
 	windowConfig->show();
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
// SLOTS MENU FORCES EXTERIEURES
/************************************************************************/
//************************************************************************/
void Window::addForceConstante()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
 	WindowConfiguration_ForceExt_Constante *windowConfig = new WindowConfiguration_ForceExt_Constante(glWidget);
 	windowConfig->show();	
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::addForceTrochoide()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
 	WindowConfiguration_ForceExt_Trochoide *windowConfig = new WindowConfiguration_ForceExt_Trochoide(glWidget);
 	windowConfig->show();	
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::addForceSoliton()
{
 System *S = glWidget->getSystem();
 if(S!=NULL) {
 	WindowConfiguration_ForceExt_Soliton *windowConfig = new WindowConfiguration_ForceExt_Soliton(glWidget);
 	windowConfig->show();	
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
}
//************************************************************************/
// SLOTS MENU DISPLAY
//************************************************************************/
//************************************************************************/
void Window::modeDisplay_Particles_POINT()
{
 glWidget->getDisplay()->setModeParticles(System::POINTS);
}
//************************************************************************/
void Window::modeDisplay_Particles_SPHERE()
{
 glWidget->getDisplay()->setModeParticles(System::SPHERES);
}
//************************************************************************/
void Window::modeDisplay_Particles_SURFACE()
{
 System *S = glWidget->getSystem();
 if(S!=NULL){
	// glWidget->getDisplay()->setDrawSurface(true);
	glWidget->getDisplay()->displaySurface();
 }
 else 
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
}
//************************************************************************/
void Window::modeDisplay_Particles_FIELD_DENSITY()
{
 System *S = glWidget->getSystem();
 if(S!=NULL)
	glWidget->displayParticlesByField(0);
}
//************************************************************************/
void Window::modeDisplay_Particles_FIELD_PRESSURE()
{
 System *S = glWidget->getSystem();
 if(S!=NULL)
	glWidget->displayParticlesByField(1);
}
//************************************************************************/
void Window::modeDisplay_Particles_FIELD_VISCOSITY()
{
 System *S = glWidget->getSystem();
 if(S!=NULL)
	glWidget->displayParticlesByField(2);
}
//************************************************************************/
void Window::modeDisplay_Particles_FIELD_STENSION()
{
 System *S = glWidget->getSystem();
 if(S!=NULL)
	glWidget->displayParticlesByField(3);
}
//************************************************************************/
void Window::modeDisplay_Particles_FIELD_TEMPERATURE()
{
 System *S = glWidget->getSystem();
 if(S!=NULL){
	if(typeid(*S)==typeid(MSPHSystem))
		glWidget->displayParticlesByField(4);
	else
		alertBox("Type d'affichage supporté que pour un MSPHSystem !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::modeDisplay_Particles_FIELD_CONCENTRATION()
{
 System *S = glWidget->getSystem();
 if(S!=NULL){
	if(typeid(*S)==typeid(MSPHSystem))
		glWidget->displayParticlesByField(6);
	else
		alertBox("Type d'affichage supporté que pour un MSPHSystem !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::modeDisplay_Particles_FIELD_MASS()
{
 System *S = glWidget->getSystem();
 if(S!=NULL){
	if(typeid(*S)==typeid(MSPHSystem))
		glWidget->displayParticlesByField(5);
	else
		alertBox("Type d'affichage supporté que pour un MSPHSystem !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::modeDisplay_Particles_FIELD_LEVEL()
{
 /*System *S = glWidget->getSystem();
 if(S!=NULL){
	if(typeid(*S)==typeid(HSPHSystem))
		glWidget->displayParticlesByField(5);
	else
		alertBox("Type d'affichage supporté que pour un MSPHSystem !!",QMessageBox::Critical);
 }*/
}
//************************************************************************/
void Window::modeDisplay_Collisions_POINT()
{
 glWidget->getDisplay()->setModeRasterization(GL_POINT);
}
//************************************************************************/
void Window::modeDisplay_Collisions_LINES()
{
 glWidget->getDisplay()->setModeRasterization(GL_LINE);
}
//************************************************************************/
void Window::modeDisplay_Collisions_FILL()
{
 glWidget->getDisplay()->setModeRasterization(GL_FILL);
}
//************************************************************************/
void Window::modeDisplay_Collisions_normales(bool b)
{
 glWidget->getDisplay()->setDrawNormales_ObjCollision(b);
}
//************************************************************************/
void Window::modeDisplay_DrawEmitters(bool b)
{
 glWidget->getDisplay()->setDrawEmitters(b);
}
//************************************************************************/
// SLOTS MENU EXTRAS
//************************************************************************/
//************************************************************************/
void Window::createAnimatedHeightField()
{
  WindowConfiguration_AnimatedHeightField *windowConfig = new WindowConfiguration_AnimatedHeightField(glWidget);
  windowConfig->show();
}
//************************************************************************/
// SLOTS MENU ANIMATION
//************************************************************************/
//************************************************************************/
void Window::play()
{
 if(glWidget->getSystem()!=NULL || glWidget->getAnimatedHF()!=NULL){
	printf("fonction animate\n");
 	glWidget->startAnimation();
 	glWidget->animate();
 }
 else {
	alertBox("Pas de de système à animer !!",QMessageBox::Critical);
 }
}
//************************************************************************/
void Window::stop()
{
	glWidget->stopAnimation();
}
//************************************************************************/
void Window::captureFrames()
{
	glWidget->captureVideo(true);
}

//************************************************************************/
// SLOT MENU RENDER
//************************************************************************/
void Window::render()
{
System *S = glWidget->getSystem();
 if(S!=NULL) {
        	current_dir = "sortie/";
 		QString filename = QFileDialog::getSaveFileName(this, tr("Save scene (mitsuba format)"), "*.xml");
		SceneExporter_Mitsuba exporter;
    		exporter._export(filename.toStdString().c_str(),S,Vector3(1.0,0.0,0.0));
		QString command1 = "mitsuba "+filename;
		system(command1.toStdString().c_str());
                QString command2 = "eog "+filename;
		command2.replace("xml","png");
		system(command2.toStdString().c_str());
 }
 else {
	        alertBox("Pas de système à exporter !!",QMessageBox::Critical);
 }

	//RLite rlite;
	//rlite.rayTrace();
/*	QPixmap image;
	image.load("/home/emma/Bureau/Travaux_en_cours/PSB_Project/Ressources/Rendering/output/img.JPG");
	QLabel *label = new QLabel();
    	label->setPixmap(image);
    	label->move(30, 20);
	const QPixmap* im = label->pixmap();
	printf("depth:%d\n",im->depth());
	label->show();*/
}
//************************************************************************/
// SLOT ACTION HELP
//************************************************************************/
void Window::help()
{
}

//************************************************************************/
//************************************************************************/
void Window::alertBox(QString text, QMessageBox::Icon icon)
{
	QMessageBox Q;
	Q.setIcon(icon);
	Q.setText(text);
        Q.exec();
}
//************************************************************************/
// FENETRE DE DIALOGUE POUR L'IMPORT
//************************************************************************/
string Window::getOpenFileName(const QString & caption,
                               const QStringList & filters,
                               int * ind_filter)
{
   QString filename;
   QFileDialog open_dialog(0, caption, current_dir.path());

   if (ind_filter != NULL) *ind_filter = -1;
   open_dialog.setFilters(filters);
   open_dialog.setAcceptMode(QFileDialog::AcceptOpen);
   open_dialog.setFileMode(QFileDialog::ExistingFile);

   if (open_dialog.exec())
   {
      filename = open_dialog.selectedFiles().at(0);
      current_dir = open_dialog.directory();

      if (ind_filter != NULL)
      {
         for ((*ind_filter) = 0;
               filters.at(*ind_filter) != open_dialog.selectedFilter();
               (*ind_filter)++);
      }
      return filename.toStdString();
   }
   return std::string("");
}
//************************************************************************/
//************************************************************************/
