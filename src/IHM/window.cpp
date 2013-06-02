#include <window.h>

#include <windowConfiguration_SimpleSystem.h>
#include <windowConfiguration_CudaSystem.h>
#include <windowConfiguration_SPHSystem.h>
#include <windowConfiguration_PCI_SPHSystem.h>
#include <windowConfiguration_MSPHSystem.h>

#include <windowConfiguration_Sphere.h>
#include <windowConfiguration_Box.h>
#include <windowConfiguration_Mesh.h>
#include <windowConfiguration_Plan.h>
#include <windowConfiguration_Cylinder.h>
#include <windowConfiguration_LinearHeightField.h>
#include <windowConfiguration_GaussianHeightField.h>
#include <windowConfiguration_PeriodicHeightField.h>
#include <windowConfiguration_CombinedHeightField.h>

#include <windowConfiguration_Emitter_Box.h>
#include <windowConfiguration_Emitter_Elipsoide.h>
#include <windowConfiguration_Emitter_Mesh.h>
#include <windowConfiguration_Emitter_Cylinder.h>
#include <windowConfiguration_Emitter_Sphere.h>
#include <windowConfiguration_Emitter_Girly.h>

#include <windowConfiguration_ForceExt_Trochoide.h>
#include <windowConfiguration_ForceExt_Constante.h>

#include <windowConfiguration_AnimatedHeightField.h>

#include <ParticleExporter_Txt.h>
#include <ParticleExporter_XML.h>
#include <ParticleExporter_Mitsuba.h>

#include <ParticleLoader_Txt.h>
#include <ParticleLoader_XML.h>

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
    export_particles_xml_act = new QAction(tr("&XML"),this);
    connect(export_particles_xml_act,SIGNAL(triggered()), this, SLOT(export_particles_xml()));
    menuExportParticles->addAction(export_particles_xml_act);
    export_particles_Mitsuba_act = new QAction(tr("&Mitsuba (vol)"),this);
    connect(export_particles_Mitsuba_act,SIGNAL(triggered()), this, SLOT(export_particles_Mitsuba()));
    menuExportParticles->addAction(export_particles_Mitsuba_act);
    menuExport->addMenu(menuExportParticles);

    menuExportSurface = new QMenu(tr("Surface"),this);
    export_surface_obj_act = new QAction(tr("&OBJ"),this);
    connect(export_surface_obj_act,SIGNAL(triggered()), this, SLOT(export_surface_obj()));
    menuExportSurface->addAction(export_surface_obj_act);
    menuExport->addMenu(menuExportSurface);

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

    simpleSystem_act = new QAction(tr("&Simple System"),this);
    connect(simpleSystem_act, SIGNAL(triggered()), this, SLOT(simpleSystem()));
    menuParticles->addAction(simpleSystem_act);

    particleInteraction_act = new QAction(tr("&CUDA System"),this);
    connect(particleInteraction_act, SIGNAL(triggered()), this, SLOT(particleInteractionSystem()));
    menuParticles->addAction(particleInteraction_act);

    fluidSystem_act = new QAction(tr("SPH System"),this);
    connect(fluidSystem_act, SIGNAL(triggered()), this, SLOT(fluidSystem()));
    menuParticles->addAction(fluidSystem_act);

    PCI_fluidSystem_act = new QAction(tr("PCISPH System"),this);
    connect(PCI_fluidSystem_act, SIGNAL(triggered()), this, SLOT(PCI_fluidSystem()));
    menuParticles->addAction(PCI_fluidSystem_act);
 
    Mixing_fluidSystem_act = new QAction(tr("MSPH System"),this);
    connect(Mixing_fluidSystem_act, SIGNAL(triggered()), this, SLOT(Mixing_fluidSystem()));
    menuParticles->addAction(Mixing_fluidSystem_act);

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

   /* createEmitterElipsoide_act = new QAction(tr("Elipsoide"),this);
    connect(createEmitterElipsoide_act, SIGNAL(triggered()), this, SLOT(createEmitterElipsoide()));
    menuEmitters->addAction(createEmitterElipsoide_act);
*/
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

   // MENU AIDE
    menuHelp = new QMenu("&Help");
    help_act = new QAction(tr("&Display"),this);
    help_act->setShortcut(tr("H"));
    connect(help_act,SIGNAL(triggered()), this, SLOT(help()));
    menuHelp->addAction(help_act);
   
   // AJOUT DES MENUS AU MENU BAR DE LA FENETRE
    menuBar->addMenu(menuFile);
    menuBar->addMenu(menuParticles);
    menuBar->addMenu(menuCollision);
    menuBar->addMenu(menuEmitters);
    menuBar->addMenu(menuForcesExt);
    menuBar->addMenu(menuDisplay);
    menuBar->addMenu(menuExtras);
    menuBar->addMenu(menuAnimation);
    menuBar->addMenu(menuHelp);

    mainLayout = new QVBoxLayout();
    mainLayout->addWidget(menuBar);
    layout2 = new QHBoxLayout();
    layout2->addWidget(glWidget);
    mainLayout->addLayout(layout2);
    setLayout(mainLayout);

    setWindowTitle(tr("PSB Project"));
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
        	current_dir = "sortie/";
 		QString filename = QFileDialog::getSaveFileName(this, tr("Save particles"), "*.txt");
		ParticleExporter_Txt exporter;
    		exporter._export(filename.toStdString().c_str(),S);
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
        	current_dir = "sortie/";
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
        	current_dir = "sortie/";
 		QString filename = QFileDialog::getSaveFileName(this, tr("Save particles"), "*.xml");
		ParticleExporter_Mitsuba exporter;
    		exporter._export(filename.toStdString().c_str(),S);
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
 		QString filename = QFileDialog::getSaveFileName(this, tr("Save particles"), "*.obj");
		glWidget->getSurface()->exportOBJ(filename.toStdString().c_str());
 }
 else {
	        alertBox("Pas de surface à exporter !!",QMessageBox::Critical);
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
void Window::fluidSystem()
{
   WindowConfiguration_SPHSystem *windowConfig = new WindowConfiguration_SPHSystem(glWidget);
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
 if(S!=NULL) {
	if(typeid(*(glWidget->getSystem()))==typeid(SPHSystem)){
		glWidget->getSurface()->extract((SPHSystem*)S,0.5,20);
	}
 }
 else {
	alertBox("Vous devez construire un système de particules avant d'effectuer cette opération !!",QMessageBox::Critical);
 }
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
 //if(glWidget->getSystem()!=NULL){
 	glWidget->startAnimation();
 	glWidget->animate();
 /*}
 else {
	alertBox("Pas de de système à animer !!",QMessageBox::Critical);
 }*/
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

