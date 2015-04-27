#include <GL/glew.h>
#include "glWidget.h"
#include <iostream>
#include <unistd.h>
#include <ParticleLoader_XML.h>
#include <ForceExt_Soliton.h>
#include <ForceExt_TrainSoliton.h>
#include <stdlib.h>
#include <time.h>

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
void GLWidget::setSurface(SurfaceSPH *S)
{
	surface = S;
}
//******************************************************************************
//******************************************************************************
unsigned int GLWidget::getFrameCourante()
{
	return frame_courante;
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
	displayByField = false;
	field = -1;
	frame_courante = 0;
	system = NULL;
        AHF = NULL;
	surface = NULL;
	display = new GLDisplay(this);
       	setAxisIsDrawn(false);
	setFPSIsDisplayed(false);
	setTextIsEnabled (true);
	//setGridIsDrawn();
	setShortcut(ANIMATION, Qt::CTRL + Qt::Key_A);
	camera()->setType(Camera::PERSPECTIVE);
	camera()->setSceneRadius(1.0);
	camera()->showEntireScene();

	initRender();

	setAnimationPeriod(1);
	setFPSIsDisplayed(true);
	
	initConfig();
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
void GLWidget::keyPressEvent( QKeyEvent *e )
{
  const char* c= e->text().toStdString().c_str();
  if ((c[0]=='i'))
		initConfig();
}

//******************************************************************************
//******************************************************************************
void GLWidget::initConfig()
{
	if(system!=NULL) { removeAll(); }
  	else {
		// CONFIG SOLITON

		nbSoliton=1;
		srand(time(NULL));
    		SPHSystem *S = new SPHSystem();
	
		S->setMinS(Vector3(-2,-1.0,-1.0));
		S->setMaxS(Vector3(2,1.0,1.0));
    		ForceExt_Constante *FG = new ForceExt_Constante(Vector3(0,-1,0),9.81);
		Vector3 domainMin = Vector3(-2.0,-0.1,-1.0);
		Vector3 domainMax = Vector3(2.0,0.4,1.0);
		//float xShoaling = -0.5; 
		//float xBreaking = 0.5;
		float xShoaling1 = domainMin.x();
		l = 1.3;//nbSoliton*0.1;
		float xBreaking1 = xShoaling1 + l;

		ForceExt_Soliton* FS1 = new ForceExt_Soliton(0.25, xShoaling1, xBreaking1, domainMin, domainMax);
		ForceExt_TrainSoliton *FST = new ForceExt_TrainSoliton();
		FST->addSoliton(FS1);

		float xShoaling2 = -0.5;
		float xBreaking2 = xShoaling2 + l;
		ForceExt_Soliton* FS2 = new ForceExt_Soliton(0.2, xShoaling2, xBreaking2, domainMin, domainMax);
		FS2->setStep(0.25);
		FST->addSoliton(FS2);

        S->addForce(FG);
		S->addForce(FST);
        
		system = S;
		ParticleLoader_XML loader;
        vector<Particle*> particles = loader.load(system,"Ressources/tunnel2.xml");
    	//	system->init(particles);
		// Config velocities with soliton model RO98 
		// For comparison
/*		float g = 9.81;
		vector<Particle*> p;
		for(uint i=0;i<system->particles.size();i++){
			SPHParticle *P = (SPHParticle*)system->particles[i];
			Vector3 pos = P->getNewPos();
			if(pos.x()<-1.0){
			float H = 6; float d = 10;
			double HD = sqrt((3*H)/(4*d*d*d))*(pos.x()+1.0);
			double sech2 = (1/(cosh(HD)*cosh(HD)));
			double tanh1 = sinh(HD)/cosh(HD);
			double u = 0.4*sqrt(g*d)*(H/d)*(1/sech2);
			double v = 20*sqrt(3*g*d)*powf(H/d,1.5)*(pos.y()/d)*sech2*tanh1;
			H = 0.3; d = 0.1;
			HD = sqrt((3*H)/(4*d*d*d))*pos.x();
			sech2 = (1/(cosh(HD)*cosh(HD)));
			tanh1 = sinh(HD)/cosh(HD);
			float y = d + H*sech2;
			/*float H = 0.32; float d = 0.1;
			 float A = sqrt(3*H/(4*d*d*d));
			 float prof = fabs(pos.y() +0.1);
			 float lambda1 = 2.5; float lambda2 = 32.5;
	 		 float u = 10*-sqrt(g*d)*powf(prof/d,3)*tanh(A*(-prof))*(1/cosh(A*(-prof)));//*lambda1;
			 float v = 10*H*powf(1/cosh(A*(pos.x()+1.0)),2);//*lambda2;
*/
			//P->setOldPos(Vector3(pos.x(),y-0.2,pos.z()));
			//P->setNewPos(P->getOldPos());
/*			P->setOldVel(Vector3(u,v,0));
			P->setNewVel(P->getOldVel());
			P->setVelInterAv(Vector3(u,v,0));
			P->setVelInterAp(Vector3(u,v,0));
			//printf("y av:%f ap:%f vel:%f %f\n",pos.y(),P->getNewPos().y(),u,v);
			}
			p.push_back(P);
		}
		system->init(p);*/
		BoxCollision *B = new BoxCollision();
		B->create(Vector3(0,0,0),4.0,0.5,0.5,0.45,true);
		system->addObjectCollision(B);
		
		//displayParticlesByField(0);
		draw();
		//animate();*/

		// CONFIG MSPH
/*		MSPHSystem *S = new MSPHSystem();
		ForceExt_Constante *FG = new ForceExt_Constante(Vector3(0,-1,0),9.81);
		S->addForce(FG);
		system = S;
		SimulationData_MSPHSystem* data1 = new SimulationData_MSPHSystem(0.002,0.002,Vector3(1,1,1),998.29,3.5,0.0728,3,//Vector3(0.37,0.26,0.23),998.29,3.5,0.0728,3,
		   20,40,Vector3(0,0,0),Vector3(0,0,0),Vector3(0,0,0));
		SimulationData_MSPHSystem* data2 = new SimulationData_MSPHSystem(0.002,0.002,Vector3(1,1,1),998.29,10,3,3,//Vector3(0.90,0.88,0.78),1000,36,6,5,
		   20,20,Vector3(0,0,0),Vector3(0,0,0),Vector3(0,0,0));
		SimulationData_MSPHSystem* data3 = new SimulationData_MSPHSystem(0.002,0.002,Vector3(1,1,1),1000,36,6,5,//Vector3(0.90,0.88,0.78),1000,36,6,5,
		   20,20,Vector3(0,0,0),Vector3(0,0,0),Vector3(0,0,0));
		EmitterElipsoide* E1 = new EmitterElipsoide(Vector3(0.0,0.2,0),0.1, 0, -1, 0, 1000,Vector3(0.0,-4.0,0));
		E1->setData(data1);
		EmitterElipsoide* E2 = new EmitterElipsoide(Vector3(-0.5,0.2,0),0.1, -1, 0, 0, 500,Vector3(-3.5,-4.0,0));
		E2->setData(data2);
		EmitterElipsoide* E3 = new EmitterElipsoide(Vector3(0.0,0.3,0),0.1, 0, -1, 0, 500,Vector3(0,-8.0,0));
		E3->setData(data3);
		BoxCollision *B = new BoxCollision();
		B->create(Vector3(0,0,0),1.0,1.0,1.0,0.45,true);
		system->addObjectCollision(B);
		system->addEmitter(E1);
		//system->addEmitter(E2);
		//system->addEmitter(E3);
		//displayParticlesByField(7);
		draw();*/
	}
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
void GLWidget::displayParticlesByField(uint field)
{
   displayByField = true;
   this->field = field;
}
//******************************************************************************
//******************************************************************************
void GLWidget::animate()
{
 if(frame_courante==2){
	ForcesExt* forces = system->getForcesExt();
	ForceExt_TrainSoliton *FST = (ForceExt_TrainSoliton*) forces->getForce(1);
	FST->getSoliton(0)->setTime(1.0);
	FST->getSoliton(1)->setTime(1.0);
	startAnimation();
 }
 /*if(frame_courante%600==0 && frame_courante>0){
	ForcesExt* forces = system->getForcesExt();
	ForceExt_Soliton *FS = (ForceExt_Soliton *) ((ForceExt_TrainSoliton*) forces->getForce(1))->getSoliton(0);
	Vector3 domainMin = FS->getDomainMin();
	Vector3 domainMax = FS->getDomainMax();
	//float xShoaling = (rand()/(double)RAND_MAX) * (domainMax.x()-domainMin.x()) + domainMin.x();
	//float l = (rand()/(double)RAND_MAX)* (1.2-0.5) + 0.5;
	float XShoaling = -1.0;//domainMin.x();
	//l += 0.1;
	float XBreaking = XShoaling + l;
	FS->setXShoaling(XShoaling);
	FS->setXBreaking(XBreaking);
	FS->setTime(1.0);
	ForceExt_TrainSoliton *FST = (ForceExt_TrainSoliton*) forces->getForce(1);
	FST->setSoliton(FS,0);
	nbSoliton++;
 }*/
 /*if(frame_courante==20){
	SceneExporter_Mitsuba exporter;
	exporter._export("/home/emmanuelle/Bureau/SOL20.xml",system,Vector3(1.0,0.0,0.0));
 }
 if(frame_courante==88){
	SceneExporter_Mitsuba exporter;
	exporter._export("/home/emmanuelle/Bureau/SOL88.xml",system,Vector3(1.0,0.0,0.0));
 }
 if(frame_courante==180){
	SceneExporter_Mitsuba exporter;
	exporter._export("/home/emmanuelle/Bureau/SOL180.xml",system,Vector3(1.0,0.0,0.0));
 }
 if(frame_courante==361){
	SceneExporter_Mitsuba exporter;
	exporter._export("/home/emmanuelle/Bureau/SOL361.xml",system,Vector3(1.0,0.0,0.0));
 }*/

 if(system!=NULL) {
	//double totalStep = 0;
	//while(totalStep<0.005) {
		system->update();
	//	totalStep += system->getDt();
	//}
 }
 if(AHF!=NULL) AHF->update();
 if(displayByField) display->displayParticlesByField(field);
 /*if(frame_courante==600){
	        SimulationData_MSPHSystem* data2 = new SimulationData_MSPHSystem(0.04,0.04,Vector3(0,0,1),1000,36,6,5,
		 40,20,Vector3(0,0,0),Vector3(0,0,0),Vector3(0,0,0));
		EmitterElipsoide* E2 = new EmitterElipsoide(Vector3(-0.5,0.2,0),0.1, 1, 0, 0, 100,Vector3(2.5,-1.0,0));
		E2->setData(data2);
		system->addEmitter(E2);
		displayParticlesByField(2);
 }*/
// printf("frame:%d\n",frame_courante);
 /**********************************************************************************************/
 // export AHF collision OBJ
 /*QString dir = "sortie/mitsuba/OBJ/AHF/";
 QString dir2 = "../OBJ/AHF/";
 QString fileName = "AHF";
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
 fileName+=str;
 fileName+=".obj";
 QString filename = dir+fileName;
 QString filenameOBJ = dir2+fileName;
 vector<ObjectCollision*> objects = system->getObjectsCollision();
 for(uint i=0;i<objects.size();i++){
	ObjectCollision* o = objects[i];
	if(typeid(*o)==typeid(AnimatedPeriodic_HeightFieldCollision)){
		AnimatedPeriodic_HeightFieldCollision* H = (AnimatedPeriodic_HeightFieldCollision*) o;
		H->exportToOBJ_noN(filename.toStdString().c_str());
	}
 }*/
 /**********************************************************************************************/
 //if(captureImage)
	//captureMitsubaSequence("");//filenameOBJ.toStdString().c_str());
	//captureImagesSequence();
 frame_courante++;

/* if(frame_courante>700){
	stopAnimation();
	printf("STOP Animation \n");
 }*/

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
   QString imageName = "sortie/Video/image";
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
}

//******************************************************************************
//******************************************************************************
void GLWidget::captureMitsubaSequence(const char* filename)
{
 System *S = getSystem();
 if(S!=NULL) {
   	QString fileName = "sortie/mitsuba/file/filePCISPH/file";
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
   	fileName+=str;
   	fileName+=".xml";
	SceneExporter_Mitsuba exporter;
    	exporter._export(fileName.toStdString().c_str(),S, Vector3(0,0,0));
 }
}
//******************************************************************************
//******************************************************************************
