#include <windowConfiguration_Emitter_Girly.h>
#include <typeinfo>

/***********************************************************************************/
/***********************************************************************************/
WindowConfiguration_Emitter_Girly::WindowConfiguration_Emitter_Girly(GLWidget *widget)
{
	QTabWidget* onglets = new QTabWidget(this);
    	onglets->setGeometry(300, 200, 240, 160);

	QWidget *page1 = new QWidget();
	QWidget *page2 = new QWidget();
	onglets->addTab(page1,"Configuration");
	onglets->addTab(page2,"Particles parameters");

	velocityLabel = new QLabel("Velocity particles",page1);
	VX = new QDoubleSpinBox(page1);
	VY = new QDoubleSpinBox(page1);
	VZ = new QDoubleSpinBox(page1);
	VX->setMinimum(-10); VX->setMaximum(10);
	VY->setMinimum(-10); VY->setMaximum(10);
	VZ->setMinimum(-10); VZ->setMaximum(10);
	VX->setValue(0.0); VY->setValue(0.0); VZ->setValue(0.0);
	connect(VX, SIGNAL(valueChanged(double)), this, SLOT(displaySphere(double))); 
	connect(VY, SIGNAL(valueChanged(double)), this, SLOT(displaySphere(double))); 
	connect(VZ, SIGNAL(valueChanged(double)), this, SLOT(displaySphere(double))); 

	QVBoxLayout *layout1 = new QVBoxLayout();
	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(velocityLabel,0,0);
	grid5->addWidget(VX,0,1);
	grid5->addWidget(VY,0,2);
	grid5->addWidget(VZ,0,3);
	layout1->addLayout(grid5);

	page1->setLayout(layout1);
	
	if(typeid(*(widget->getSystem()))==typeid(SimpleSystem))
		configData = new WindowConfiguration_Data_SimpleSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(SPHSystem))
		configData = new WindowConfiguration_Data_SPHSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(CudaSystem))
		configData = new WindowConfiguration_Data_CudaSystem(page2);

	if(typeid(*(widget->getSystem()))==typeid(PCI_SPHSystem))
		configData = new WindowConfiguration_Data_PCI_SPHSystem(page2);

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

	B = new EmitterGirly(Vector3(VX->value(),VY->value(),VZ->value()));
	B->setData(configData->getData());

	this->glWidget = widget;
	this->glWidget->getDisplay()->displayEmitter(B);
}
/***********************************************************************************/
WindowConfiguration_Emitter_Girly::~WindowConfiguration_Emitter_Girly()
{
}
/***********************************************************************************/
/***********************************************************************************/
void WindowConfiguration_Emitter_Girly::accept()
{
	System *system = glWidget->getSystem();
	B->setData(configData->getData());
	system->addEmitter(B);
	system->emitParticles();
	close();
}
/***********************************************************************************/
void WindowConfiguration_Emitter_Girly::cancel()
{
	delete(B);
	close();
}
/***********************************************************************************/
/***********************************************************************************/
EmitterGirly* WindowConfiguration_Emitter_Girly::getEmitter()
{
	return B;
}
/***********************************************************************************/
/***********************************************************************************/
void WindowConfiguration_Emitter_Girly::displayEmitter(double d)
{
	if(B==NULL) delete(B);
	B = new EmitterGirly(Vector3(VX->value(),VY->value(),VZ->value()));
	B->setData(configData->getData());
	glWidget->getDisplay()->displayEmitter(B);
}
/***********************************************************************************/
/***********************************************************************************/
