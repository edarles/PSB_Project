#include <windowConfiguration_Emitter_Elipsoide.h>
/*********************************************************************************************/
/*********************************************************************************************/
WindowConfiguration_Emitter_Elipsoide::WindowConfiguration_Emitter_Elipsoide(GLWidget *widget)
{
	QTabWidget* onglets = new QTabWidget(this);
    	onglets->setGeometry(300, 200, 240, 160);

	QWidget *page1 = new QWidget();
	QWidget *page2 = new QWidget();
	onglets->addTab(page1,"Configuration");
	onglets->addTab(page2,"Particles parameters");

	originLabel = new QLabel("Center",page1);
	OX = new QDoubleSpinBox(page1);
	OY = new QDoubleSpinBox(page1);
	OZ = new QDoubleSpinBox(page1);
	OX->setMinimum(-100); OX->setMaximum(100);
	OY->setMinimum(-100); OY->setMaximum(100);
	OZ->setMinimum(-100); OZ->setMaximum(100);
	OX->setValue(0.75); OY->setValue(0.2); OZ->setValue(0.0);
	connect(OX, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 
	connect(OY, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 
	connect(OZ, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 

	radiusLabel = new QLabel("Radius",page1);
 	radius = new QDoubleSpinBox(page1);
	radius->setValue(0.10);
	connect(radius, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 
	
	directionLabel = new QLabel("Direction",page1);
	DX = new QDoubleSpinBox(page1);
	DY = new QDoubleSpinBox(page1);
	DZ = new QDoubleSpinBox(page1);
	DX->setMinimum(-1.0); DX->setMaximum(1.0);
	DY->setMinimum(-1.0); DY->setMaximum(1.0);
	DZ->setMinimum(-1.0); DZ->setMaximum(1.0);
	DX->setValue(-1.0); DY->setValue(0.0); DZ->setValue(0.0);
	connect(DX, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 
	connect(DY, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 
	connect(DZ, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 

	durationTimeLabel = new QLabel("Duration Time",page1);
 	durationTime = new QSpinBox(page1);
	durationTime->setMinimum(1);
	durationTime->setMaximum(1000);
	durationTime->setValue(300.0);
	connect(durationTime, SIGNAL(valueChanged(int)), this, SLOT(displayElipsoide(int))); 

	velocityLabel = new QLabel("Velocity particles",page1);
	VX = new QDoubleSpinBox(page1);
	VY = new QDoubleSpinBox(page1);
	VZ = new QDoubleSpinBox(page1);
	VX->setMinimum(-100); VX->setMaximum(100);
	VY->setMinimum(-100); VY->setMaximum(100);
	VZ->setMinimum(-100); VZ->setMaximum(100);
	VX->setValue(-2.5); VY->setValue(-1.0); VZ->setValue(0.0);
	connect(VX, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 
	connect(VY, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 
	connect(VZ, SIGNAL(valueChanged(double)), this, SLOT(displayElipsoide(double))); 

	QVBoxLayout *layout1 = new QVBoxLayout();
	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(originLabel,0,0);
	grid1->addWidget(OX,0,1); 
	grid1->addWidget(OY,0,2);
	grid1->addWidget(OZ,0,3);
	layout1->addLayout(grid1);

	QGridLayout *grid8 = new QGridLayout();
	grid8->addWidget(directionLabel,0,0);
	grid8->addWidget(DX,0,1); 
	grid8->addWidget(DY,0,2);
	grid8->addWidget(DZ,0,3);
	layout1->addLayout(grid8);

	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(radiusLabel,0,0);
	grid2->addWidget(radius,0,1);
	layout1->addLayout(grid2);

	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(durationTimeLabel,0,0);
	grid5->addWidget(durationTime,0,1);
	layout1->addLayout(grid5);
	
	QGridLayout *grid6 = new QGridLayout();
	grid6->addWidget(velocityLabel,0,0);
	grid6->addWidget(VX,0,1);
	grid6->addWidget(VY,0,2);
	grid6->addWidget(VZ,0,3);
	layout1->addLayout(grid6);

	page1->setLayout(layout1);

	if(typeid(*(widget->getSystem()))==typeid(SimpleSystem))
		configData = new WindowConfiguration_Data_SimpleSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(CudaSystem))
		configData = new WindowConfiguration_Data_CudaSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(SPHSystem))
		configData = new WindowConfiguration_Data_SPHSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(PCI_SPHSystem))
		configData = new WindowConfiguration_Data_PCI_SPHSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(WCSPHSystem))
		configData = new WindowConfiguration_Data_WCSPHSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(MSPHSystem))
		configData = new WindowConfiguration_Data_MSPHSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(SWSPHSystem))
		configData = new WindowConfiguration_Data_SWSPHSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(SPH2DSystem))
		configData = new WindowConfiguration_Data_SPH2DSystem(page2);

	QGridLayout *grid7 = new QGridLayout();
	buttonOK = new QPushButton(tr("OK"),this);
	connect(buttonOK, SIGNAL(clicked()), this , SLOT(accept()));
	buttonCancel = new QPushButton(tr("Cancel"),this);
        connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
	grid7->addWidget(buttonOK,0,0);
	grid7->addWidget(buttonCancel,0,1);

	layout = new QVBoxLayout();
	layout->addWidget(onglets);
	layout->addLayout(grid7);

	setLayout(layout);
	setWindowTitle("Configuration");
	B = new EmitterElipsoide(Vector3(OX->value(),OY->value(),OZ->value()),radius->value(), DX->value(), DY->value(), DZ->value(),
					 durationTime->value(),Vector3(VX->value(),VY->value(),VZ->value()));
	B->setData(configData->getData());
	this->glWidget = widget;
	this->glWidget->getDisplay()->displayEmitter(B);
}
/*********************************************************************************************/
WindowConfiguration_Emitter_Elipsoide::~WindowConfiguration_Emitter_Elipsoide()
{
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Emitter_Elipsoide::accept()
{
	System *system = glWidget->getSystem();
	B->setData(configData->getData());
	system->addEmitter(B);
	printf("addEmitter\n");
	system->emitParticles();
	close();
}
/*********************************************************************************************/
void WindowConfiguration_Emitter_Elipsoide::cancel()
{
	delete(B);
	close();
}
/*********************************************************************************************/
/*********************************************************************************************/
EmitterElipsoide* WindowConfiguration_Emitter_Elipsoide::getEmitter()
{
	return B;
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Emitter_Elipsoide::displayElipsoide(double d)
{
	if(DX->value()==1 || DX->value()==-1) { DY->setValue(0); DZ->setValue(0);}
	if(DY->value()==1 || DY->value()==-1) { DX->setValue(0); DZ->setValue(0);}
	if(DZ->value()==1 || DZ->value()==-1) { DX->setValue(0); DY->setValue(0);}

	if(B==NULL) delete(B);
	B = new EmitterElipsoide(Vector3(OX->value(),OY->value(),OZ->value()),radius->value(),DX->value(), DY->value(), DZ->value(),
					 durationTime->value(),Vector3(VX->value(),VY->value(),VZ->value()));
	B->setData(configData->getData());
	glWidget->getDisplay()->displayEmitter(B);
}
/*********************************************************************************************/
void WindowConfiguration_Emitter_Elipsoide::displayElipsoide(int d)
{
	if(B==NULL) delete(B);
	B = new EmitterElipsoide(Vector3(OX->value(),OY->value(),OZ->value()),radius->value(),DX->value(), DY->value(), DZ->value(),
					 durationTime->value(),Vector3(VX->value(),VY->value(),VZ->value()));
	B->setData(configData->getData());
	glWidget->getDisplay()->displayEmitter(B);
}
/*********************************************************************************************/
/*********************************************************************************************/
